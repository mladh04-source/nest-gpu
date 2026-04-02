# -*- coding: utf-8 -*-
#
# nest_gpu_code_generator.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.
import copy
import glob
import os
import shutil
from typing import Dict, Sequence, Optional, Mapping, Any, List

from pynestml.codegeneration.printers.c_simple_expression_printer import CSimpleExpressionPrinter
from pynestml.codegeneration.printers.cpp_printer import CppPrinter
from pynestml.codegeneration.printers.cpp_expression_printer import CppExpressionPrinter
from pynestml.codegeneration.printers.nest_gpu_function_call_printer import NESTGPUFunctionCallPrinter
from pynestml.codegeneration.printers.nest_gpu_numeric_function_call_printer import NESTGPUNumericFunctionPrinter
from pynestml.codegeneration.printers.nest_gpu_numeric_variable_printer import NESTGPUNumericVariablePrinter
from pynestml.codegeneration.printers.nest_gpu_variable_printer import NESTGPUVariablePrinter
from pynestml.codegeneration.printers.unitless_c_simple_expression_printer import UnitlessCSimpleExpressionPrinter
from pynestml.meta_model.ast_model import ASTModel
from pynestml.utils.logger import LoggingLevel, Logger
from pynestml.codegeneration.nest_code_generator import NESTCodeGenerator
from pynestml.frontend.frontend_configuration import FrontendConfiguration


def replace_text_between_tags(filepath, replace_str, begin_tag="// <<BEGIN_NESTML_GENERATED>>",
                              end_tag="// <<END_NESTML_GENERATED>>", rfind=False):
    with open(filepath, "r") as f:
        file_str = f.read()

    if rfind:
        start_pos = file_str.rfind(begin_tag) + len(begin_tag)
        end_pos = file_str.rfind(end_tag)
    else:
        start_pos = file_str.find(begin_tag) + len(begin_tag)
        end_pos = file_str.find(end_tag)

    file_str = file_str[:start_pos] + replace_str + file_str[end_pos:]
    with open(filepath, "w") as f:
        f.write(file_str)
    f.close()


class NESTGPUCodeGenerator(NESTCodeGenerator):
    """
    A code generator for NEST GPU target
    """

    _default_options = {
    "neuron_parent_class": "BaseNeuron",
    "neuron_parent_class_include": "archiving_node.h",
    "preserve_expressions": False,
    "simplify_expression": "sympy.logcombine(sympy.powsimp(sympy.expand(expr)))",
    "neuron_models": [],
    "synapse_models": [],
    "neuron_synapse_pairs": [],
    "continuous_state_buffering_method": "continuous_time_buffer",
    "gsl_adaptive_step_size_controller": "with_respect_to_solution",
    "gap_junctions": {"enable": False},
    "templates": {
        "path": "resources_nest_gpu/point_neuron",
        "model_templates": {
            "neuron": ["@NEURON_NAME@.cu.jinja2", "@NEURON_NAME@.h.jinja2"]
        },
        "module_templates": []
    },
    "solver": "analytic",
    "numeric_solver": "rk45",
    "nest_gpu_path": None,
    "nest_gpu_build_path": "build",
    "nest_gpu_install_path": "install",
    "experimental_templates": {
        "path": "resources_nest_gpu_experimental/point_neuron",
        "model_templates": {
            "neuron": ["@NEURON_NAME@.cu.jinja2", "@NEURON_NAME@.h.jinja2"]
        },
        "module_templates": []
    }
}

    def __init__(self, options: Optional[Mapping[str, Any]] = None):
        merged_options = copy.deepcopy(NESTCodeGenerator._default_options)
        self._deep_update(merged_options, copy.deepcopy(NESTGPUCodeGenerator._default_options))

        if options:
            self._deep_update(merged_options, dict(options))

        # NEST-GPU currently supports neuron templates only.
        # Remove synapse/module templates inherited from NESTCodeGenerator defaults.
        merged_options["templates"]["model_templates"] = {
            "neuron": ["@NEURON_NAME@.cu.jinja2", "@NEURON_NAME@.h.jinja2"]
        }
        merged_options["templates"]["module_templates"] = []

        merged_options["experimental_templates"]["model_templates"] = {
            "neuron": ["@NEURON_NAME@.cu.jinja2", "@NEURON_NAME@.h.jinja2"]
        }
        merged_options["experimental_templates"]["module_templates"] = []

        super(NESTCodeGenerator, self).__init__(merged_options)

        self.nest_gpu_path = None
        self.analytic_solver = {}
        self.numeric_solver = {}
        self.non_equations_state_variables = {}

        self._sync_runtime_options_from_options_dict()

        self.setup_template_env()
        self.setup_printers()
        
    @staticmethod
    def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                NESTGPUCodeGenerator._deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    @staticmethod
    def _prepare_experimental_template_tree(standard_template_path: str,
                                            experimental_template_path: str) -> None:
        base_dir = os.path.dirname(__file__)

        if not os.path.isabs(standard_template_path):
            std_path = os.path.abspath(os.path.join(base_dir, standard_template_path))
        else:
            std_path = standard_template_path

        if not os.path.isabs(experimental_template_path):
            exp_path = os.path.abspath(os.path.join(base_dir, experimental_template_path))
        else:
            exp_path = experimental_template_path

        std_directives = os.path.join(std_path, "directives")
        exp_directives = os.path.join(exp_path, "directives")

        if not os.path.isdir(exp_path):
            raise FileNotFoundError(
                f"Experimental template path does not exist: {exp_path}"
            )

        if not os.path.isdir(std_path):
            raise FileNotFoundError(
                f"Standard template path does not exist: {std_path}"
            )

        if os.path.isdir(std_directives) and not os.path.isdir(exp_directives):
            shutil.copytree(std_directives, exp_directives)
            
    def _sync_runtime_options_from_options_dict(self) -> None:
        """
        Synchronize derived runtime attributes after options have been updated.
        This is crucial because Frontend sets codegen_opts after __init__.
        """
        opt_nest_gpu_path = self.get_option("nest_gpu_path") if self.option_exists("nest_gpu_path") else None

        if opt_nest_gpu_path:
            self.nest_gpu_path = opt_nest_gpu_path
        else:
            if "NEST_GPU" in os.environ:
                self.nest_gpu_path = os.environ["NEST_GPU"]
            else:
                self.nest_gpu_path = os.getcwd()

        # keep runtime option dictionary in sync
        self._options["nest_gpu_path"] = self.nest_gpu_path

        Logger.log_message(
            None,
            -1,
            "The NEST-GPU path was set to: " + str(self.nest_gpu_path),
            None,
            LoggingLevel.INFO
        )

        solver = self.get_option("solver") if self.option_exists("solver") else "analytic"

        if solver == "experimental":
            exp_templates = copy.deepcopy(self.get_option("experimental_templates"))
            std_templates = copy.deepcopy(NESTGPUCodeGenerator._default_options["templates"])

            self._prepare_experimental_template_tree(
                standard_template_path=std_templates["path"],
                experimental_template_path=exp_templates["path"]
            )
            self._options["templates"] = exp_templates
        else:
            self._options["templates"] = copy.deepcopy(
                NESTGPUCodeGenerator._default_options["templates"]
            )

    def set_options(self, options: Mapping[str, Any]):
        """
        Override to react to frontend-provided codegen_opts after generator construction.
        """
        unused_opts = super().set_options(options)
        self._sync_runtime_options_from_options_dict()
        self.setup_template_env()
        return unused_opts

    def setup_printers(self):
        super().setup_printers()

        self._nest_variable_printer = NESTGPUVariablePrinter(
            expression_printer=None,
            with_origin=True,
            with_vector_parameter=False
        )
        self._nest_function_call_printer = NESTGPUFunctionCallPrinter(None)
        self._printer = CppExpressionPrinter(
            simple_expression_printer=CSimpleExpressionPrinter(
                variable_printer=self._nest_variable_printer,
                constant_printer=self._constant_printer,
                function_call_printer=self._nest_function_call_printer
            )
        )
        self._nest_variable_printer._expression_printer = self._printer
        self._nest_function_call_printer._expression_printer = self._printer
        self._nest_printer = CppPrinter(expression_printer=self._printer)

        self._nest_variable_printer_no_origin = NESTGPUVariablePrinter(
            None,
            with_origin=False,
            with_vector_parameter=False
        )
        self._nest_function_call_printer_no_origin = NESTGPUFunctionCallPrinter(None)
        self._printer_no_origin = CppExpressionPrinter(
            simple_expression_printer=CSimpleExpressionPrinter(
                variable_printer=self._nest_variable_printer_no_origin,
                constant_printer=self._constant_printer,
                function_call_printer=self._nest_function_call_printer_no_origin
            )
        )
        self._nest_variable_printer_no_origin._expression_printer = self._printer_no_origin
        self._nest_function_call_printer_no_origin._expression_printer = self._printer_no_origin

        self._gsl_variable_printer = NESTGPUNumericVariablePrinter(None)
        self._gsl_function_call_printer = NESTGPUNumericFunctionPrinter(None)
        self._gsl_printer = CppExpressionPrinter(
            simple_expression_printer=UnitlessCSimpleExpressionPrinter(
                variable_printer=self._gsl_variable_printer,
                constant_printer=self._constant_printer,
                function_call_printer=self._gsl_function_call_printer
            )
        )
        self._gsl_function_call_printer._expression_printer = self._gsl_printer

    def generate_module_code(self, neurons: Sequence[ASTModel], synapses: Sequence[ASTModel]):
        self.copy_models_from_target_path()
        self.add_model_name_to_neuron_header(neurons)
        self.add_model_to_neuron_class(neurons)
        self.add_files_to_makefile(neurons)

    def copy_models_from_target_path(self):
        types = ["*.h", "*.cu"]
        dst_path = os.path.join(self.nest_gpu_path, "src")
        for _type in types:
            for file in glob.glob(os.path.join(FrontendConfiguration.get_target_path(), _type)):
                shutil.copy(file, dst_path)

    def add_model_name_to_neuron_header(self, neurons: List[ASTModel]):
        neuron_models_h_path = str(os.path.join(self.nest_gpu_path, "src", "neuron_models.h"))
        shutil.copy(neuron_models_h_path, neuron_models_h_path + ".bak")

        neuron_indexes = []
        neuron_names = []
        for neuron in neurons:
            neuron_indexes.append("\ni_" + neuron.get_name() + "_model,")
            neuron_names.append("\n, \"" + neuron.get_name() + "\"")

        neuron_indexes = "".join(neuron_indexes) + "\n"
        neuron_names = "".join(neuron_names) + "\n"
        replace_text_between_tags(neuron_models_h_path, neuron_indexes)
        replace_text_between_tags(neuron_models_h_path, neuron_names, rfind=True)

    def add_model_to_neuron_class(self, neurons: List[ASTModel]):
        neuron_models_cu_path = str(os.path.join(self.nest_gpu_path, "src", "neuron_models.cu"))
        shutil.copy(neuron_models_cu_path, neuron_models_cu_path + ".bak")

        include_files = []
        code_blocks = []
        for neuron in neurons:
            include_files.append("\n#include \"" + neuron.get_name() + ".h\"")
            model_name_index = "i_" + neuron.get_name() + "_model"
            model_name = neuron.get_name()
            n_ports = len(neuron.get_spike_input_ports())
            code_blocks.append(
                "\n"
                f"else if (model_name == neuron_model_name[{model_name_index}]) {{\n"
                f"    n_ports = {n_ports};\n"
                f"    {model_name} *{model_name}_group = new {model_name};\n"
                f"    node_vect_.push_back({model_name}_group);\n"
                " }"
            )
        include_files = "".join(include_files) + "\n"
        code_blocks = "".join(code_blocks) + "\n"
        replace_text_between_tags(neuron_models_cu_path, include_files)
        replace_text_between_tags(neuron_models_cu_path, code_blocks, rfind=True)

    def add_files_to_makefile(self, neurons: Sequence[ASTModel]):
        cmakelists_path = str(os.path.join(self.nest_gpu_path, "src", "CMakeLists.txt"))
        shutil.copy(cmakelists_path, cmakelists_path + ".bak")

        gen_files = []
        for neuron in neurons:
            gen_files.append(f'    "{neuron.get_name()}.h"\n')
            gen_files.append(f'    "{neuron.get_name()}.cu"\n')
        gen_files = "".join(gen_files) + "\n"
        replace_text_between_tags(
            cmakelists_path,
            gen_files,
            begin_tag="# <<BEGIN_NESTML_GENERATED>>",
            end_tag="# <<END_NESTML_GENERATED>>"
        )

    def _get_neuron_model_namespace(self, neuron: ASTModel) -> Dict:
        namespace = super()._get_neuron_model_namespace(neuron)
        if namespace["uses_numeric_solver"]:
            namespace["printer"] = self._gsl_printer
            namespace["uses_analytic_solver"] = False
        return namespace

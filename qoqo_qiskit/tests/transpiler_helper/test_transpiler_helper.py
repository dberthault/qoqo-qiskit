# Copyright © 2023 HQS Quantum Simulations GmbH.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
"""Test file for interface.py."""

from qiskit import Aer
from qoqo import Circuit, CircuitDag, QuantumProgram
from qoqo import operations as ops
from qoqo.measurements import PauliZProduct, PauliZProductInput  # type:ignore
from qiskit.providers.fake_provider import GenericBackendV2

from qoqo_qiskit.transpiler_helper.transpiler_helper import (
    transpile_with_qiskit,
    transpile_program_with_qiskit,
)  # type:ignore


def test_basic_circuit_basic_gates() -> None:
    """Test basic circuit conversion with a BaseGates transpiler."""
    circuit = Circuit()
    circuit += ops.PauliX(0)
    circuit += ops.Identity(0)

    circuit_res = Circuit()
    circuit_res += ops.SqrtPauliX(0)
    circuit_res += ops.SqrtPauliX(0)

    transpiled_circuit = transpile_with_qiskit(circuit, [{"basis_gates": ["sx", "rz", "cz"]}])

    assert transpiled_circuit == circuit_res


def test_medium_circuit_basic_gates() -> None:
    """Test basic circuit conversion with a BaseGates transpiler."""
    circuit = Circuit()
    circuit += ops.CNOT(0, 1)

    circuit_res = Circuit()
    circuit_res += ops.RotateZ(1, 1.57079632679)
    circuit_res += ops.SqrtPauliX(1)
    circuit_res += ops.RotateZ(1, 3.14159265358)
    circuit_res += ops.ControlledPauliZ(0, 1)
    circuit_res += ops.SqrtPauliX(1)
    circuit_res += ops.RotateZ(1, 1.57079632679)

    transpiled_circuit = transpile_with_qiskit(circuit, [{"basis_gates": ["sx", "rz", "cz"]}])
    transpiled_circuit_dag = CircuitDag()
    circuit_res_dag = CircuitDag()
    transpiled_circuit_dag.from_circuit(transpiled_circuit)
    circuit_res_dag.from_circuit(circuit_res)

    assert transpiled_circuit_dag == circuit_res_dag


def test_basic_circuit_basic_backend() -> None:
    """Test basic circuit conversion with a backend transpiler."""
    circuit = Circuit()
    circuit += ops.PauliX(0)
    circuit += ops.Identity(0)

    circuit_res = Circuit()
    circuit_res += ops.PauliX(0)

    backend = GenericBackendV2(5)
    transpiled_circuit = transpile_with_qiskit(circuit, [{"backend": backend}])

    assert transpiled_circuit == circuit_res


def test_toffoli_circuit_basic_gates() -> None:
    """Test toffoli circuit conversion with a BaseGates transpiler."""
    circuit = Circuit()
    circuit += ops.Toffoli(0, 1, 2)

    circuit_res = Circuit()
    circuit_res += ops.Hadamard(2)
    circuit_res += ops.CNOT(1, 2)
    circuit_res += ops.RotateZ(2, -0.78539816339)
    circuit_res += ops.CNOT(0, 2)
    circuit_res += ops.TGate(2)
    circuit_res += ops.CNOT(1, 2)
    circuit_res += ops.TGate(1)
    circuit_res += ops.RotateZ(2, -0.78539816339)
    circuit_res += ops.CNOT(0, 2)
    circuit_res += ops.CNOT(0, 1)
    circuit_res += ops.TGate(2)
    circuit_res += ops.TGate(0)
    circuit_res += ops.RotateZ(1, -0.78539816339)
    circuit_res += ops.Hadamard(2)
    circuit_res += ops.CNOT(0, 1)

    transpiled_circuit = transpile_with_qiskit(
        circuit, [{"basis_gates": ["rz", "sx", "x", "cx", "h", "t"]}]
    )
    transpiled_circuit_dag = CircuitDag()
    circuit_res_dag = CircuitDag()
    transpiled_circuit_dag.from_circuit(transpiled_circuit)
    circuit_res_dag.from_circuit(circuit_res)

    assert transpiled_circuit_dag == circuit_res_dag


def test_toffoli_circuit_backend() -> None:
    """Test toffoli circuit conversion with a backend transpiler."""
    circuit = Circuit()
    circuit += ops.Toffoli(0, 1, 2)

    circuit_res = Circuit()
    circuit_res += ops.Hadamard(2)
    circuit_res += ops.CNOT(1, 2)
    circuit_res += ops.RotateZ(2, -0.78539816339)
    circuit_res += ops.CNOT(0, 2)
    circuit_res += ops.TGate(2)
    circuit_res += ops.CNOT(1, 2)
    circuit_res += ops.TGate(1)
    circuit_res += ops.RotateZ(2, -0.78539816339)
    circuit_res += ops.CNOT(0, 2)
    circuit_res += ops.CNOT(0, 1)
    circuit_res += ops.TGate(2)
    circuit_res += ops.TGate(0)
    circuit_res += ops.RotateZ(1, -0.78539816339)
    circuit_res += ops.Hadamard(2)
    circuit_res += ops.CNOT(0, 1)

    backend = GenericBackendV2(5)
    transpiled_circuit = transpile_with_qiskit(circuit, [{"backend": backend}])

    transpiled_circuit_dag = CircuitDag()
    circuit_res_dag = CircuitDag()
    transpiled_circuit_dag.from_circuit(transpiled_circuit)
    circuit_res_dag.from_circuit(circuit_res)

    assert transpiled_circuit_dag == circuit_res_dag


def test_multiple_circuits_backend() -> None:
    """Test multiple circuits conversion with a BaseGates transpiler."""
    circuit_1 = Circuit()
    circuit_1 += ops.PauliX(0)
    circuit_1 += ops.Identity(0)

    circuit_res_1 = Circuit()
    circuit_res_1 += ops.PauliX(0)

    circuit_2 = Circuit()
    circuit_2 += ops.Toffoli(0, 1, 2)

    circuit_res_2 = Circuit()
    circuit_res_2 += ops.Hadamard(2)
    circuit_res_2 += ops.CNOT(1, 2)
    circuit_res_2 += ops.RotateZ(2, -0.78539816339)
    circuit_res_2 += ops.CNOT(0, 2)
    circuit_res_2 += ops.TGate(2)
    circuit_res_2 += ops.CNOT(1, 2)
    circuit_res_2 += ops.TGate(1)
    circuit_res_2 += ops.RotateZ(2, -0.78539816339)
    circuit_res_2 += ops.CNOT(0, 2)
    circuit_res_2 += ops.CNOT(0, 1)
    circuit_res_2 += ops.TGate(2)
    circuit_res_2 += ops.TGate(0)
    circuit_res_2 += ops.RotateZ(1, -0.78539816339)
    circuit_res_2 += ops.Hadamard(2)
    circuit_res_2 += ops.CNOT(0, 1)

    backend = GenericBackendV2(5)
    transpiled_circuits = transpile_with_qiskit([circuit_1, circuit_2], [{"backend": backend}])

    transpiled_circuit_dag = CircuitDag()
    circuit_res_dag = CircuitDag()
    transpiled_circuit_dag.from_circuit(transpiled_circuits[1])
    circuit_res_dag.from_circuit(circuit_res_2)

    assert transpiled_circuits[0] == circuit_res_1 and transpiled_circuit_dag == circuit_res_dag


def assert_quantum_program_equal(
    quantum_program_1: QuantumProgram, quantum_program2: QuantumProgram
) -> None:
    """Assert that two quantum programs are equal.

    Args:
        quantum_program_1 (QuantumProgram): quantum program
        quantum_program2 (QuantumProgram): quantum program

    Raises:
        AssertionError: if the quantum programs are not equal
    """
    assert quantum_program_1.input_parameter_names() == quantum_program2.input_parameter_names()
    assert quantum_program_1.measurement().input() == quantum_program2.measurement().input()
    assert (
        quantum_program_1.measurement().constant_circuit()
        == quantum_program2.measurement().constant_circuit()
    )
    for circuit_1, circuit_2 in zip(
        quantum_program_1.measurement().circuits(), quantum_program2.measurement().circuits()
    ):
        circuit_dag_1 = CircuitDag()
        circuit_dag_2 = CircuitDag()
        circuit_dag_1.from_circuit(circuit_1)
        circuit_dag_2.from_circuit(circuit_2)
        assert circuit_dag_1 == circuit_dag_2


def test_basic_program_basic_gates() -> None:
    """Test basic program conversion with a BaseGates transpiler."""
    circuit_1 = Circuit()
    circuit_1 += ops.PauliX(0)
    circuit_1 += ops.Identity(0)

    circuit_res_1 = Circuit()
    circuit_res_1 += ops.SqrtPauliX(0)
    circuit_res_1 += ops.SqrtPauliX(0)

    circuit_2 = Circuit()
    circuit_2 += ops.CNOT(0, 1)

    circuit_res_2 = Circuit()
    circuit_res_2 += ops.RotateZ(1, 1.57079632679)
    circuit_res_2 += ops.SqrtPauliX(1)
    circuit_res_2 += ops.RotateZ(1, 3.14159265358)
    circuit_res_2 += ops.ControlledPauliZ(0, 1)
    circuit_res_2 += ops.SqrtPauliX(1)
    circuit_res_2 += ops.RotateZ(1, 1.57079632679)

    measurement_input = PauliZProductInput(1, False)
    measurement = PauliZProduct(
        constant_circuit=None, circuits=[circuit_1, circuit_2], input=measurement_input
    )
    measurement_res = PauliZProduct(
        constant_circuit=None, circuits=[circuit_res_1, circuit_res_2], input=measurement_input
    )
    quantum_program = QuantumProgram(measurement=measurement, input_parameter_names=["x"])
    quantum_program_res = QuantumProgram(measurement=measurement_res, input_parameter_names=["x"])

    transpiled_program = transpile_program_with_qiskit(
        quantum_program, [{"basis_gates": ["sx", "rz", "crz"]}]
    )

    assert_quantum_program_equal(transpiled_program, quantum_program_res)


def test_quantum_program_backend() -> None:
    """Test basic program conversion with a BaseGates transpiler."""
    circuit_1 = Circuit()
    circuit_1 += ops.PauliX(0)
    circuit_1 += ops.Identity(0)

    circuit_res_1 = Circuit()
    circuit_res_1 += ops.SqrtPauliX(0)
    circuit_res_1 += ops.SqrtPauliX(0)

    circuit_2 = Circuit()
    circuit_2 += ops.CNOT(0, 1)

    circuit_res_2 = Circuit()
    circuit_res_2 += ops.RotateZ(1, 1.57079632679)
    circuit_res_2 += ops.SqrtPauliX(1)
    circuit_res_2 += ops.RotateZ(1, 3.14159265358)
    circuit_res_2 += ops.ControlledPauliZ(0, 1)
    circuit_res_2 += ops.SqrtPauliX(1)
    circuit_res_2 += ops.RotateZ(1, 1.57079632679)

    circuit_3 = Circuit()
    circuit_3 += ops.Toffoli(0, 1, 2)

    circuit_res_3 = Circuit()
    circuit_res_3 += ops.Hadamard(2)
    circuit_res_3 += ops.CNOT(1, 2)
    circuit_res_3 += ops.RotateZ(2, -0.78539816339)
    circuit_res_3 += ops.CNOT(0, 2)
    circuit_res_3 += ops.TGate(2)
    circuit_res_3 += ops.CNOT(1, 2)
    circuit_res_3 += ops.TGate(1)
    circuit_res_3 += ops.RotateZ(2, -0.78539816339)
    circuit_res_3 += ops.CNOT(0, 2)
    circuit_res_3 += ops.CNOT(0, 1)
    circuit_res_3 += ops.TGate(2)
    circuit_res_3 += ops.TGate(0)
    circuit_res_3 += ops.RotateZ(1, -0.78539816339)
    circuit_res_3 += ops.Hadamard(2)
    circuit_res_3 += ops.CNOT(0, 1)

    measurement_input = PauliZProductInput(1, False)
    measurement = PauliZProduct(
        constant_circuit=None, circuits=[circuit_1, circuit_2, circuit_3], input=measurement_input
    )
    measurement_res = PauliZProduct(
        constant_circuit=None,
        circuits=[circuit_res_1, circuit_res_2, circuit_res_3],
        input=measurement_input,
    )
    quantum_program = QuantumProgram(measurement=measurement, input_parameter_names=["x"])
    quantum_program_res = QuantumProgram(measurement=measurement_res, input_parameter_names=["x"])

    backend = GenericBackendV2(5)
    transpiled_program = transpile_program_with_qiskit(quantum_program, [{"backend": backend}])

    assert_quantum_program_equal(transpiled_program, quantum_program_res)

# Copyright © 2023-2025 HQS Quantum Simulations GmbH.
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
"""Test file for utils.py."""

import pytest
import sys

from qoqo_qiskit.utils import (
    struqture_hamiltonian_to_qiskit_op,
    get_qoqo_noise_models_from_aer_noise_model,
)
from struqture_py.spins import PauliHamiltonian, PauliProduct  # type: ignore
from qiskit_aer.noise import NoiseModel, pauli_error
from qoqo.noise_models import DecoherenceOnGateModel
from struqture_py.spins import PlusMinusProduct, PlusMinusLindbladNoiseOperator


def test_basic_hamiltonian() -> None:
    """Test struqture_hamiltonian_to_qiskit_op with a basic Hamiltonian."""
    pp = PauliProduct().x(0).z(1).y(2)

    hamiltonian = PauliHamiltonian()
    hamiltonian.add_operator_product(pp, 0.5)

    res = struqture_hamiltonian_to_qiskit_op(hamiltonian, 3)

    assert res.num_qubits == 3
    assert res.to_list() == [("YZX", (0.5 + 0j))]


def test_big_hamiltonian() -> None:
    """Test struqture_hamiltonian_to_qiskit_op with a big Hamiltonian."""
    pp = PauliProduct().x(0).z(1).y(2).x(3).z(4).y(5).x(6).z(7).y(8).x(9).z(10).y(11)
    pp2 = PauliProduct().x(12)

    hamiltonian = PauliHamiltonian()
    hamiltonian.add_operator_product(pp, 0.5)
    hamiltonian.add_operator_product(pp2, 0.25)

    res = struqture_hamiltonian_to_qiskit_op(hamiltonian, 13)

    assert res.num_qubits == 13
    assert res.to_list() == [("IYZXYZXYZXYZX", (0.5 + 0j)), ("XIIIIIIIIIIII", (0.25 + 0j))]


def test_converts_single_qubit_aer_qerror_to_qoqo_noise_model():
    # Aer error: identity with 90% probability, X error with 10%.
    aer_noise_model = NoiseModel()
    aer_noise_model.add_quantum_error(
        pauli_error([("I", 0.9), ("X", 0.1)]),
        instructions=["x"],
        qubits=[0],
    )

    model = get_qoqo_noise_models_from_aer_noise_model(aer_noise_model)

    expected_noise = PlusMinusLindbladNoiseOperator()
    for op, factor in [("+", 0.5), ("-", 0.5), ("Z", 0.25)]:
        product = PlusMinusProduct().from_string(f"0{op}")
        expected_noise.add_operator_product((product, product), factor * 0.1)

    expected_model = DecoherenceOnGateModel()
    expected_model = expected_model.set_single_qubit_gate_error(
        "PauliX",
        0,
        expected_noise,
    )

    assert model == expected_model


# For pytest
if __name__ == "__main__":
    pytest.main(sys.argv)

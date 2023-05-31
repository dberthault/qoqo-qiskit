"""Test qoqo_qiskit_devices information retrieval"""
# Copyright © 2023 HQS Quantum Simulations GmbH. All Rights Reserved.
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

import pytest
import sys
import numpy as np

from qoqo_qiskit_devices import ibm_devices, set_qiskit_noise_information


def test_belem_info_update(mocked: bool):
    """Test IBMBelemDevice qiskit's info update."""
    belem = ibm_devices.IBMBelemDevice()

    assert belem.single_qubit_gate_time("PauliX", 0) == 1.0
    assert belem.two_qubit_gate_time("CNOT", 0, 1) == 1.0
    assert belem.three_qubit_gate_time("ControlledControlledPauliZ", 0, 1, 2) == None
    assert belem.multi_qubit_gate_time("MultiQubitMS", [0, 1, 2, 3]) == None
    assert np.all(belem.qubit_decoherence_rates(0) == 0.0)

    set_qiskit_noise_information(belem, get_mocked_information=True)

    assert belem.single_qubit_gate_time("PauliX", 0) != 1.0
    assert belem.two_qubit_gate_time("CNOT", 0, 1) != 1.0
    assert belem.three_qubit_gate_time("ControlledControlledPauliZ", 0, 1, 2) == None
    assert belem.multi_qubit_gate_time("MultiQubitMS", [0, 1, 2, 3]) == None
    assert np.any(belem.qubit_decoherence_rates(0) != 0.0)


if __name__ == "__main__":
    pytest.main(sys.argv)

// Copyright © 2023 HQS Quantum Simulations GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use pyo3::prelude::*;

use roqoqo_qiskit_devices::IBMQuitoDevice;

/// IBM Quito device
///
#[pyclass(name = "IBMQuitoDevice", module = "qoqo_qiskit_devices")]
#[derive(Clone, Debug, PartialEq)]
pub struct IBMQuitoDeviceWrapper {
    /// Internal storage of [roqoqo_qiskit_devices::IBMQuitoDevice]
    pub internal: IBMQuitoDevice,
}

#[pymethods]
impl IBMQuitoDeviceWrapper {
    /// Create a new IBMQuitoDevice instance.
    #[new]
    pub fn new() -> Self {
        Self {
            internal: IBMQuitoDevice::new(),
        }
    }

    /// IBM's identifier.
    ///
    /// Returns:
    ///     str: The IBM's identifier of the Device.
    pub fn name(&self) -> &str {
        roqoqo_qiskit_devices::IBMQuitoDevice::name()
    }
}

impl Default for IBMQuitoDeviceWrapper {
    fn default() -> Self {
        Self::new()
    }
}

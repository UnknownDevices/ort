use alloc::string::ToString;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct TensorRTRTXExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; TensorRTRTXExecutionProvider);

impl TensorRTRTXExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.options.set("device_id", device_id.to_string());
		self
	}

	/// # Safety
	/// The provided `stream` must outlive the environment/session created with the execution provider.
	#[must_use]
	pub unsafe fn with_compute_stream(mut self, stream: *mut ()) -> Self {
		self.options.set("user_compute_stream", (stream as usize).to_string());
		self
	}

	#[must_use]
	pub fn with_max_workspace_size(mut self, max_size: usize) -> Self {
		self.options.set("nv_max_workspace_size", max_size.to_string());
		self
	}

	#[must_use]
	pub fn with_max_shared_mem_size(mut self, max_size: usize) -> Self {
		self.options.set("nv_max_shared_mem_size", max_size.to_string());
		self
	}

	#[must_use]
	pub fn with_dump_subgraphs(mut self, enable: bool) -> Self {
		self.options.set("nv_dump_subgraphs", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_detailed_build_log(mut self, enable: bool) -> Self {
		self.options.set("nv_detailed_build_log", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_cuda_graph(mut self, enable: bool) -> Self {
		self.options.set("enable_cuda_graph", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_profile_min_shapes(mut self, min_shapes: impl ToString) -> Self {
		self.options.set("nv_profile_min_shapes", min_shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_max_shapes(mut self, max_shapes: impl ToString) -> Self {
		self.options.set("nv_profile_max_shapes", max_shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_opt_shapes(mut self, opt_shapes: impl ToString) -> Self {
		self.options.set("nv_profile_opt_shapes", opt_shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_multi_profile_enable(mut self, enable: bool) -> Self {
		self.options.set("nv_multi_profile_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_use_external_data_initializer(mut self, enable: bool) -> Self {
		self.options.set("nv_use_external_data_initializer", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_runtime_cache_path(mut self, path: impl ToString) -> Self {
		self.options.set("nv_runtime_cache_path", path.to_string());
		self
	}
}

impl ExecutionProvider for TensorRTRTXExecutionProvider {
	fn name(&self) -> &'static str {
		"NvTensorRTRTXExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")), all(target_os = "windows", target_arch = "x86_64")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "tensorrt_rtx"))]
		{
			use core::ptr;

			use crate::{AsPointer, ortsys, util};

			let ffi_options = self.options.to_ffi();

			let provider_name = std::ffi::CString::new(self.name()).unwrap();
			ortsys![unsafe SessionOptionsAppendExecutionProvider(session_builder.ptr_mut(), provider_name.as_ptr(), ffi_options.key_ptrs(), ffi_options.value_ptrs(), ffi_options.len())?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}


#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>
#include <optional>

at::Tensor rms_norm_rope(at::Tensor &x, c10::optional<at::Tensor>& weights_, at::Tensor &cos_freqs, at::Tensor &sin_freqs, bool out_16bit);

at::Tensor fp6_pack(at::Tensor &x);
at::Tensor fp6_unpack(at::Tensor &x, int64_t original_n);

at::Tensor rms_norm_split_rope(
    at::Tensor &x,
    at::Tensor &sin_freqs,
    at::Tensor &cos_freqs,
    at::Tensor &weights,
    bool out_fp8
);

namespace {

at::Tensor py_rms_norm_rope(const at::Tensor &x, const std::optional<at::Tensor> &weights, const at::Tensor &cos_freqs, const at::Tensor &sin_freqs, bool out_16bit) {
    at::Tensor x_mut = x;
    at::Tensor cos_mut = cos_freqs;
    at::Tensor sin_mut = sin_freqs;
    c10::optional<at::Tensor> w = weights.has_value() ? c10::optional<at::Tensor>(*weights) : c10::nullopt;
    return rms_norm_rope(x_mut, w, cos_mut, sin_mut, out_16bit);
}

at::Tensor py_fp6_pack(const at::Tensor &x) {
    at::Tensor x_mut = x;
    return fp6_pack(x_mut);
}

at::Tensor py_fp6_unpack(const at::Tensor &x, int64_t original_n) {
    at::Tensor x_mut = x;
    return fp6_unpack(x_mut, original_n);
}

at::Tensor py_rms_norm_split_rope(const at::Tensor &x, const at::Tensor &sin_freqs, const at::Tensor &cos_freqs, const at::Tensor &weights, bool out_fp8) {
    at::Tensor x_mut = x;
    at::Tensor sin_mut = sin_freqs;
    at::Tensor cos_mut = cos_freqs;
    at::Tensor w_mut = weights;
    return rms_norm_split_rope(x_mut, sin_mut, cos_mut, w_mut, out_fp8);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_rope", &py_rms_norm_rope,
          "fused norm + rope + cvt",
          py::arg("x"), py::arg("weights") = std::nullopt, py::arg("cos_freqs"), py::arg("sin_freqs"), py::arg("out_16bit"));
    m.def("fp6_pack", &py_fp6_pack,
          "Pack 8-bit to 6-bit by dropping e_1 and e_2 bits",
          py::arg("x"));
    m.def("fp6_unpack", &py_fp6_unpack,
          "Unpack 6-bit to 8-bit (with e_1 and e_2 set to 0)",
          py::arg("x"), py::arg("original_n"));
    m.def("rms_norm_split_rope", &py_rms_norm_split_rope,
          "RMS norm + split RoPE with optional FP8 output",
          py::arg("x"), py::arg("sin_freqs"), py::arg("cos_freqs"), py::arg("weights"), py::arg("out_fp8"));
}

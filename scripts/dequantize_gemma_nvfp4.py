"""
Dequantize Gemma 4 12B NVFP4 safetensors to native BF16 for LTX-2.5.
Uses the Blackwell RTX 5080 NVFP4 kernel to ultra-fast dequantize (seconds)
instead of downloading 26GB over the network.
"""

import time
from pathlib import Path
import safetensors.torch as st
import torch
import ltx_kernels.nvfp4 as nvfp4

def main():
    src = Path("models/ltx25/text_encoders/gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors")
    dst = Path("models/ltx25/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")

    print(f"[*] Opening {src}...")
    f = st.safe_open(str(src), framework="pt")
    metadata = f.metadata() or {}
    keys = f.keys()
    print(f"[*] Total keys in source: {len(keys)}")

    # Group quantized weights
    quant_keys = set()
    for k in keys:
        if k.endswith(".weight_scale_2"):
            base_weight = k[:-len(".weight_scale_2")] + ".weight"
            quant_keys.add(base_weight)

    print(f"[*] Identified {len(quant_keys)} NVFP4 quantized weight tensors.")

    output_tensors = {}
    skipped_companion_keys = 0
    dequant_count = 0
    t0 = time.time()

    for idx, k in enumerate(keys):
        if k.endswith(".weight_scale") or k.endswith(".weight_scale_2") or k.endswith(".comfy_quant"):
            skipped_companion_keys += 1
            continue

        if k in quant_keys:
            # NVFP4 dequantize
            base = k[:-len(".weight")]
            w = f.get_tensor(k).cuda()
            s = f.get_tensor(f"{base}.weight_scale").cuda()
            s2 = f.get_tensor(f"{base}.weight_scale_2").cuda()
            
            dq = nvfp4.dequantize_nvfp4(w, s2, s, out_dtype=torch.bfloat16)
            output_tensors[k] = dq.cpu()
            dequant_count += 1
            if dequant_count % 50 == 0 or dequant_count == len(quant_keys):
                print(f"    Dequantized {dequant_count}/{len(quant_keys)} weights ({time.time() - t0:.1f}s)...")
        else:
            # Plain tensor (norm, bias, scalar, projection, etc.)
            t = f.get_tensor(k)
            output_tensors[k] = t

    print(f"[*] Dequantization completed in {time.time() - t0:.2f}s!")
    print(f"[*] Total tensors to save: {len(output_tensors)}")
    print(f"[*] Saving to {dst} (this may take ~10-20 seconds to stream to disk)...")
    
    t1 = time.time()
    st.save_file(output_tensors, str(dst), metadata=metadata)
    print(f"[+] Saved {dst} ({dst.stat().st_size / (1024**3):.2f} GB) in {time.time() - t1:.2f}s!")

if __name__ == "__main__":
    main()

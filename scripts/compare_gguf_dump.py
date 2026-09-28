#!/usr/bin/env python3
"""Compares tools/dump's output against an independent parse of the same
GGUF file via the Python `gguf` library, item by item. This is the
automated check PROMPTS.md's stage 0 requires ("与 Python gguf 库输出逐项
比对(脚本自动比对,不靠肉眼)") -- it never asks a human to eyeball two
dumps side by side.

Usage:
    python3 scripts/compare_gguf_dump.py <model.gguf> [--dump-bin PATH]
Exit code 0 iff every KV entry, array, and tensor matches; otherwise prints
every mismatch found and exits 1.
"""
import argparse
import subprocess
import sys

import gguf


def unescape(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n == "n":
                out.append("\n")
            elif n == "t":
                out.append("\t")
            elif n == "r":
                out.append("\r")
            elif n == "\\":
                out.append("\\")
            else:
                out.append(c)
                out.append(n)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def run_dump(dump_bin: str, model_path: str) -> dict:
    proc = subprocess.run([dump_bin, model_path], capture_output=True, text=True, check=True)
    lines = proc.stdout.split("\n")
    result = {"version": None, "alignment": None, "kv": {}, "tensors": {}}
    for line in lines:
        if not line:
            continue
        fields = line.split("\t")
        tag = fields[0]
        if tag == "GGUF_VERSION":
            result["version"] = int(fields[1])
        elif tag == "ALIGNMENT":
            result["alignment"] = int(fields[1])
        elif tag == "N_KV" or tag == "N_TENSORS":
            pass
        elif tag == "KV":
            _idx, key, type_name, value = fields[1], fields[2], fields[3], fields[4]
            result["kv"][key] = ("scalar", type_name, unescape(value))
        elif tag == "ARR":
            _idx, key, elem_type, n = fields[1], fields[2], fields[3], int(fields[4])
            elems = [unescape(x) for x in fields[5:]]
            assert len(elems) == n, f"ARR {key}: expected {n} elements, got {len(elems)}"
            result["kv"][key] = ("array", elem_type, elems)
        elif tag == "TENSOR":
            _idx, name, type_name, n_dims = fields[1], fields[2], fields[3], int(fields[4])
            ne = [int(x) for x in fields[5:9]]
            nbytes = int(fields[9])
            result["tensors"][name] = (type_name, n_dims, ne, nbytes)
        else:
            raise ValueError(f"unrecognized dump line tag: {tag!r}")
    return result


def float_close(a: str, b) -> bool:
    fa = float(a)
    fb = float(b)
    return abs(fa - fb) <= abs(fb) * 1e-6 + 1e-9


def compare(model_path: str, dump_bin: str) -> int:
    mine = run_dump(dump_bin, model_path)
    ref = gguf.GGUFReader(model_path)

    failures = []

    if mine["version"] != int(ref.fields["GGUF.version"].parts[-1][0]):
        failures.append(f"version mismatch: mine={mine['version']} ref={ref.fields['GGUF.version']}")

    ref_align = 32
    align_field = ref.fields.get("general.alignment")
    if align_field is not None:
        ref_align = int(align_field.parts[-1][0])
    if mine["alignment"] != ref_align:
        failures.append(f"alignment mismatch: mine={mine['alignment']} ref={ref_align}")

    ref_kv_keys = [k for k in ref.fields.keys() if not k.startswith("GGUF.")]
    if set(mine["kv"].keys()) != set(ref_kv_keys):
        only_mine = set(mine["kv"].keys()) - set(ref_kv_keys)
        only_ref = set(ref_kv_keys) - set(mine["kv"].keys())
        if only_mine:
            failures.append(f"KV keys only in mine: {sorted(only_mine)}")
        if only_ref:
            failures.append(f"KV keys only in ref: {sorted(only_ref)}")

    for key in ref_kv_keys:
        field = ref.fields[key]
        is_array = field.types[0] == gguf.GGUFValueType.ARRAY
        if key not in mine["kv"]:
            continue
        kind, type_name, value = mine["kv"][key]
        if is_array:
            elem_type = field.types[1]
            if kind != "array":
                failures.append(f"{key}: expected array, dump gave scalar")
                continue
            if type_name != elem_type.name:
                failures.append(f"{key}: elem type mismatch mine={type_name} ref={elem_type.name}")
                continue
            n = len(field.data)
            if len(value) != n:
                failures.append(f"{key}: array length mismatch mine={len(value)} ref={n}")
                continue
            for j in range(n):
                part = field.parts[field.data[j]]
                if elem_type == gguf.GGUFValueType.STRING:
                    ref_val = bytes(part).decode("utf-8")
                    if value[j] != ref_val:
                        failures.append(f"{key}[{j}]: mine={value[j]!r} ref={ref_val!r}")
                elif elem_type in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
                    if not float_close(value[j], part[0]):
                        failures.append(f"{key}[{j}]: mine={value[j]} ref={part[0]}")
                elif elem_type == gguf.GGUFValueType.BOOL:
                    ref_val = "true" if bool(part[0]) else "false"
                    if value[j] != ref_val:
                        failures.append(f"{key}[{j}]: mine={value[j]} ref={ref_val}")
                else:
                    if int(value[j]) != int(part[0]):
                        failures.append(f"{key}[{j}]: mine={value[j]} ref={int(part[0])}")
                if len(failures) > 50:
                    failures.append(f"... (stopping after 50 mismatches in {key})")
                    break
        else:
            elem_type = field.types[0]
            if kind != "scalar":
                failures.append(f"{key}: expected scalar, dump gave array")
                continue
            if type_name != elem_type.name:
                failures.append(f"{key}: type mismatch mine={type_name} ref={elem_type.name}")
                continue
            ref_val = field.parts[-1]
            if elem_type == gguf.GGUFValueType.STRING:
                ref_str = bytes(ref_val).decode("utf-8")
                if value != ref_str:
                    failures.append(f"{key}: mine={value!r} ref={ref_str!r}")
            elif elem_type in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
                if not float_close(value, ref_val[0]):
                    failures.append(f"{key}: mine={value} ref={ref_val[0]}")
            elif elem_type == gguf.GGUFValueType.BOOL:
                ref_str = "true" if bool(ref_val[0]) else "false"
                if value != ref_str:
                    failures.append(f"{key}: mine={value} ref={ref_str}")
            else:
                if int(value) != int(ref_val[0]):
                    failures.append(f"{key}: mine={value} ref={int(ref_val[0])}")

    ref_tensor_names = {t.name for t in ref.tensors}
    if set(mine["tensors"].keys()) != ref_tensor_names:
        only_mine = set(mine["tensors"].keys()) - ref_tensor_names
        only_ref = ref_tensor_names - set(mine["tensors"].keys())
        if only_mine:
            failures.append(f"tensors only in mine: {sorted(only_mine)}")
        if only_ref:
            failures.append(f"tensors only in ref: {sorted(only_ref)}")

    block_size_bytes = gguf.GGML_QUANT_SIZES
    for t in ref.tensors:
        if t.name not in mine["tensors"]:
            continue
        type_name, n_dims, ne, nbytes = mine["tensors"][t.name]
        if type_name.upper() != t.tensor_type.name:
            failures.append(f"{t.name}: type mismatch mine={type_name} ref={t.tensor_type.name}")
            continue
        ref_dims = list(t.shape)
        if n_dims != len(ref_dims):
            failures.append(f"{t.name}: n_dims mismatch mine={n_dims} ref={len(ref_dims)}")
        else:
            for d in range(n_dims):
                if ne[d] != int(ref_dims[d]):
                    failures.append(
                        f"{t.name}: ne[{d}] mismatch mine={ne[d]} ref={int(ref_dims[d])}"
                    )
        block_size, type_size = block_size_bytes[t.tensor_type]
        n_elems = 1
        for d in ref_dims:
            n_elems *= int(d)
        ref_nbytes = n_elems * type_size // block_size
        if nbytes != ref_nbytes:
            failures.append(f"{t.name}: nbytes mismatch mine={nbytes} ref={ref_nbytes}")

    print(f"KV entries checked: {len(ref_kv_keys)}")
    print(f"Tensors checked: {len(ref_tensor_names)}")
    if failures:
        print(f"FAIL: {len(failures)} mismatch(es)")
        for f in failures[:100]:
            print("  -", f)
        return 1
    print("PASS: dump matches python gguf library exactly")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a .gguf file")
    ap.add_argument("--dump-bin", default="build/tools/mi_dump")
    args = ap.parse_args()
    sys.exit(compare(args.model, args.dump_bin))

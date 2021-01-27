import sys
import os
import re


def rename_defined_constants(text):
    # #define GB_PUBLIC extern
    text = text.replace("GB_PUBLIC", "extern")

    return text


def remove_directives(text):
    # There are a few cases of safe `#define` directives that we need to keep
    #  - #define FOO 12
    #  - #define BAR ...
    safe_define = re.compile(r"^#define\s+\w+\s+(\d+|\.{3})\s*$")

    out = []
    multiline = False
    for line in text.splitlines():
        if not line:
            out.append(line)
        elif multiline:
            if line[-1] != "\\":
                multiline = False
            out.append(f"/* {line} */")
        elif line.lstrip()[0] == "#":
            if line[-1] == "\\":
                multiline = True
            if not multiline and safe_define.match(line):
                out.append(line)
            else:
                out.append(f"/* {line} */")
        else:
            out.append(line)
    return "\n".join(out)


def remove_complex(text):
    out = []
    extern_block = None
    complex_extern = False
    for line in text.splitlines():
        if extern_block is not None:
            if "FC32" in line or "FC64" in line:
                complex_extern = True
            if line.replace(" ", "") == ");":
                # End of extern block
                if complex_extern:
                    extern_block[0] = f"/* {extern_block[0]}"
                    extern_block.append(f"{line} */")
                else:
                    extern_block.append(line)
                out.extend(extern_block)
                extern_block = None
                complex_extern = False
            else:
                extern_block.append(line)
        elif not line:
            out.append(line)
        elif line.strip() == "extern":
            extern_block = [line]
        elif ("FC32" in line or "FC64" in line) and line[:2] != "/*":
            out.append(f"/* {line} */")
        else:
            out.append(line)
    return "\n".join(out)


def main(filename):
    with open(filename, "r") as f:
        text = f.read()
    text = rename_defined_constants(text)
    text = remove_directives(text)
    if "_no_complex_" in filename:
        text = remove_complex(text)
    with open(filename, "w") as f:
        f.write(text)


if __name__ == "__main__":
    filename = sys.argv[1]
    if not os.path.exists(filename):
        raise Exception(f'"{filename}" does not exist')
    main(filename)

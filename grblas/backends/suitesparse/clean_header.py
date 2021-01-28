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
    safe_define = re.compile(r"^#define\s+\w+\s+(\d+|\.{3})(\s|$)")

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
            if line.replace(" ", "") == ");":  # End of extern block
                extern_block.append(line)
                if complex_extern:
                    for i in range(len(extern_block)):
                        extern_block[i] = f"// {extern_block[i]}"
                out.extend(extern_block)
                extern_block = None
                complex_extern = False
            else:
                extern_block.append(line)
        elif not line:
            out.append(line)
        elif line.strip() == "extern":
            extern_block = [line]
        elif "FC32" in line or "FC64" in line:
            # Check if the line is a terminating line
            if re.search(r"FC(32|64)\s*;", line):
                # By commenting the terminating line out, we lose the closing semicolon
                # Walk up the lines, looking for the last non-commented out line
                # then replace the trailing comma with a semicolon
                for i in range(5):  # it's never more than 5 away
                    if out[-i].startswith("//"):
                        continue
                    last_comma_pos = out[-i].rfind(",")
                    if last_comma_pos < 0:
                        continue
                    out[-i] = f"{out[-i][:last_comma_pos]};{out[-i][last_comma_pos+1:]}"
                    break
            out.append(f"// {line}")
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

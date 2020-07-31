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
    safe_define = re.compile(r"^#define\s+\S+\s+(\d+|\.{3})$")

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


def main(filename):
    with open(filename, "r") as f:
        text = f.read()
    text = rename_defined_constants(text)
    text = remove_directives(text)
    with open(filename, "w") as f:
        f.write(text)


if __name__ == "__main__":
    filename = sys.argv[1]
    if not os.path.exists(filename):
        raise Exception(f'"{filename}" does not exist')
    main(filename)

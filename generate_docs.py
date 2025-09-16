"""
Parses Python source files to generate a single markdown document of API
reference documentation.
"""
import ast
import sys


LIVE_BASE_URL = "https://github.com/matomatical/matthewplotlib/blob/main"


def main(
    paths: list[str],
    /,
    code_base_url: str = LIVE_BASE_URL,
):
    for path in paths:
        if path.endswith(".md"):
            print(f"echoing {path}...", file=sys.stderr)
            with open(path) as f:
                print(f.read())
        
        elif path.endswith(".py"):
            print(f"processing {path}...", file=sys.stderr)
            # parse
            with open(path) as f:
                source_code = f.read()
            tree = ast.parse(source_code)
            # generate
            visitor = MarkdownVisitor(path, code_base_url)
            visitor.visit(tree)
            print("\n".join(visitor.markdown))
        
        else:
            print(f"skipping {path}...", file=sys.stderr)

    print("done.", file=sys.stderr)


class MarkdownVisitor(ast.NodeVisitor):
    """
    Traverses an Abstract Syntax Tree and extracts docstrings
    from modules, classes, and functions to build a Markdown document.
    """
    def __init__(
        self,
        path: str,
        code_base_url: str,
    ):
        module = path[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            self.module = module[:-len(".__init__")]
        else:
            self.module = module
        self.path = path
        self.code_url = "/".join([code_base_url, path])
        self.markdown: list[str] = []
        self.classctx: None | str = None


    def visit_Module(self, node: ast.Module):
        # section for the module
        name = s(self.module)
        self.markdown.append(f"## module {name}\n")
    
        # path and source link
        self.markdown.append(
            f"**{s(self.path)}** "
            f"([source]({self.code_url}))\n"
        )
    
        # docstring
        module_docstring = ast.get_docstring(node)
        if module_docstring:
            self.markdown.append(f"{module_docstring}\n")
    
        # children
        self.generic_visit(node)
    

    def visit_ClassDef(self, node: ast.ClassDef):
        # skip private classes
        if node.name.startswith('_'):
            return

        # subsection for the class
        if self.classctx:
            # inner class
            name = f'{self.classctx}.{s(node.name)}'
        else:
            name = s(node.name)
        self.markdown.append(f"### class {name}\n")
        
        # inheritance and source link
        bases = ', '.join(
            [s(ast.unparse(b)) for b in node.bases if b is not None]
        )
        self.markdown.append(
            f"**{name}({bases}):** "
            f"([source]({self.code_url}#L{node.lineno}))\n"
        )

        # docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self.markdown.append(f"{docstring}\n")
            
        # children
        oldctx = self.classctx
        self.classctx = name
        self.generic_visit(node)
        self.classctx = oldctx

        # end class with a horizontal divider
        self.markdown.append("---\n")


    def visit_FunctionDef(self, node: ast.FunctionDef):
        # skip private methods (but not dunder methods)
        if node.name.startswith('_') and not node.name.startswith('__'):
            return
        # (but *do* skip __init__ methods)
        if node.name == '__init__':
            return

        # section for the method or function
        name = s(node.name)
        if self.classctx is not None:
            self.markdown.append(f"### method {self.classctx}.{name}\n")
        else:
            self.markdown.append(f"### function {name}\n")

        # signature and source link
        args_str = ", ".join([
            arg.arg
            + f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
            for arg in node.args.args
        ])
        return_str = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        signature = f"{name}({s(args_str)}){s(return_str)}"
        self.markdown.append(
            f"**{signature}:** "
            f"([source]({self.code_url}#L{node.lineno}))\n"
        )
        
        # docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self.markdown.append(f"{docstring}\n")

        # no recursion to children of methods or functions...

        # end function with a horizontal divider
        if self.classctx is None:
            self.markdown.append("---\n")


    def visit_TypeAlias(self, node: ast.TypeAlias):
        # skip private methods
        if node.name.id.startswith('_'):
            return

        # section for the type alias
        name = s(node.name.id)
        self.markdown.append(f"### type {name}\n")

        # TODO: allow a docstring somehow...

        # title and source link
        self.markdown.append(
            f"**{s(ast.unparse(node))}** "
            f"([source]({self.code_url}#L{node.lineno}))\n"
        )


def s(s: str) -> str:
    """
    Sanitise string for markdown.
    """
    return s.replace("_", r"\_").replace("*", r"\*")


if __name__ == "__main__":
    main(sys.argv[1:])

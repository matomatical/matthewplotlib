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
            print(f"parsing {path}...", file=sys.stderr)
            with open(path) as f:
                source_code = f.read()
            tree = ast.parse(source_code)
            module = path[:-3].replace("/", ".")
            
            print(f"processing {module}...", file=sys.stderr)
            code_url = "/".join([code_base_url, path])
            visitor = MarkdownVisitor(module, code_url)
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
        module: str,
        code_url: str,
    ):
        if module.endswith(".__init__"):
            self.module = module[:-len(".__init__")]
        else:
            self.module = module
        self.code_url = code_url
        self.markdown: list[str] = []
        self.context: list[str] = []


    def visit_Module(self, node: ast.Module):
        # section for the module
        name = s(self.module)
        self.markdown.append(f"## module {name}\n")
    
        # source link
        self.markdown.append(f"[[source]({self.code_url})]\n")
    
        # docstring
        module_docstring = ast.get_docstring(node)
        if module_docstring:
            self.markdown.append(f"{module_docstring}\n")
    
        # children
        self.context.append(name)
        self.generic_visit(node)
        self.context.pop()
    

    def visit_ClassDef(self, node: ast.ClassDef):
        # skip private classes
        if node.name.startswith('_'):
            return

        # subsection for the class
        context = '.'.join(self.context)
        name = s(node.name)
        self.markdown.append(f"### class {context}.{name}\n")
        
        # source link
        self.markdown.append(f"[[source]({self.code_url}#L{node.lineno})]\n")
        
        # base classes
        bases = ', '.join(
            [s(ast.unparse(b)) for b in node.bases if b is not None]
        )
        if bases:
            inheritance_str = f"[Inherits from {bases}]"
            self.markdown.append(f"{inheritance_str}\n")

        # docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self.markdown.append(f"{docstring}\n")
            
        # children
        self.context.append(name)
        self.generic_visit(node)
        self.context.pop()


    def visit_FunctionDef(self, node: ast.FunctionDef):
        # skip private methods (but not dunder methods)
        if node.name.startswith('_') and not node.name.startswith('__'):
            return
        # (but *do* skip __init__ methods)
        if node.name == '__init__':
            return

        # section for the method or function
        context = '.'.join(self.context)
        name = s(node.name)
        if len(self.context) > 1:
            self.markdown.append(f"### method {context}.{name}\n")
        else:
            self.markdown.append(f"### function {context}.{name}\n")

        # signature
        args_str = ", ".join([
            arg.arg
            + f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
            for arg in node.args.args
        ])
        return_str = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        signature = f"{name}({s(args_str)}){s(return_str)}"
        self.markdown.append(f"**{signature}**\n")
        
        # source link
        self.markdown.append(f"[[source]({self.code_url}#L{node.lineno})]\n")
        
        # docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self.markdown.append(f"{docstring}\n")

        # no recursion to children of methods or functions


    def visit_TypeAlias(self, node: ast.TypeAlias):
        # skip private methods
        if node.name.id.startswith('_'):
            return

        # section for the type alias
        context = '.'.join(self.context)
        name = s(node.name.id)
        self.markdown.append(f"### type {context}.{name}\n")

        self.markdown.append(f"**{s(ast.unparse(node))}**\n")

        # source link
        self.markdown.append(f"[[source]({self.code_url}#L{node.lineno})]\n")


def s(s: str) -> str:
    """
    Sanitise string for markdown.
    """
    return s.replace("_", r"\_").replace("*", r"\*")


if __name__ == "__main__":
    main(sys.argv[1:])

import urllib.parse


def main():
    code = """\
import piplite
await piplite.install(["pyhf==0.7.0rc4", "requests"])
%matplotlib inline
import pyhf\
"""

    parsed_url = urllib.parse.quote(code)
    url_base = "https://jupyterlite.github.io/demo/repl/index.html"
    jupyterlite_options = "?kernel=python&toolbar=1&code="
    jupyterlite_url = url_base + jupyterlite_options + parsed_url

    print(f"# jupyterlite URL:\n{jupyterlite_url}")

    jupyterlite_iframe_rst = f"""\
   <iframe
      src="{jupyterlite_url}"
      width="100%"
      height="500px"
   ></iframe>\
"""
    print(f"\n# RST for iframe for jupyterlite.rst:\n{jupyterlite_iframe_rst}")


if __name__ == "__main__":
    raise SystemExit(main())

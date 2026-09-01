"""Export the derived fixed-head/no-hand 27DoF model next to its mesh tree."""

from __future__ import annotations

from pathlib import Path

from wbc_mjlab.casbot02ampj import constants as C


DEFAULT_OUTPUT = C.SOURCE_XML.with_name("CASBOT02AMPJ_27dof_gmr.xml")


def main(output: Path = DEFAULT_OUTPUT) -> None:
  output = output.expanduser().resolve()
  spec = C.get_spec()
  output.write_text(spec.to_xml(), encoding="utf-8")
  model = spec.compile()
  print(
    f"[OK] {output}: hinges={model.njnt - 1}, nq={model.nq}, "
    f"nv={model.nv}, bodies={model.nbody - 1}"
  )


if __name__ == "__main__":
  main()

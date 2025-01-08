import xmltodict

from pathlib import Path
from typing import Any
from ._metadata import CellVoyagerAcquisition


def _parse(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return xmltodict.parse(
            f.read(),
            process_namespaces=True,
            namespaces={"http://www.yokogawa.co.jp/BTS/BTSSchema/1.0": None},
            attr_prefix="",
            cdata_key="Value",
        )


def load_wpi(wpi_path: Path) -> CellVoyagerAcquisition:
    if not wpi_path.exists():
        raise FileNotFoundError(f"{wpi_path} does not exist.")
    wpi_dict = _parse(wpi_path)
    mlf_dict = _parse(wpi_path.parent / "MeasurementData.mlf")
    mrf_dict = _parse(wpi_path.parent / "MeasurementDetail.mrf")
    mes_path = (
        wpi_path.parent / mrf_dict["MeasurementDetail"]["MeasurementSettingFileName"]
    )
    mes_dict = _parse(mes_path)
    return CellVoyagerAcquisition(
        **wpi_dict,
        **mlf_dict,
        **mrf_dict,
        **mes_dict,
    )

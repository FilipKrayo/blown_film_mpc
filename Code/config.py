"""
config.py
=========
Central configuration module for the Co-Extrusion Blown Film Line
MPC project.

All physical constants, dataset column definitions, signal role
assignments, and tunable hyperparameters are defined here so that
no magic numbers or string literals appear elsewhere in the codebase.

Author : Blown Film MPC Project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Dataset column registry
# ---------------------------------------------------------------------------

ALL_COLUMNS: List[str] = [
    "Datum",
    "ST0_VARActAuftrag",

    # Extruder 0 — Heating Zones 3-9
    "ST110_VARExtr_0_HeizungZone_3_Regler_Y",
    "ST110_VARExtr_0_HeizungZone_3_SollTemp",
    "ST110_VARExtr_0_HeizungZone_4_Konfig",
    "ST110_VARExtr_0_HeizungZone_4_Regler_X",
    "ST110_VARExtr_0_HeizungZone_4_Regler_Y",
    "ST110_VARExtr_0_HeizungZone_4_SollTemp",
    "ST110_VARExtr_0_HeizungZone_5_Konfig",
    "ST110_VARExtr_0_HeizungZone_5_Regler_X",
    "ST110_VARExtr_0_HeizungZone_5_Regler_Y",
    "ST110_VARExtr_0_HeizungZone_5_SollTemp",
    "ST110_VARExtr_0_HeizungZone_6_Konfig",
    "ST110_VARExtr_0_HeizungZone_6_Regler_X",
    "ST110_VARExtr_0_HeizungZone_6_Regler_Y",
    "ST110_VARExtr_0_HeizungZone_6_SollTemp",
    "ST110_VARExtr_0_HeizungZone_7_Konfig",
    "ST110_VARExtr_0_HeizungZone_7_Regler_X",
    "ST110_VARExtr_0_HeizungZone_7_Regler_Y",
    "ST110_VARExtr_0_HeizungZone_7_SollTemp",
    "ST110_VARExtr_0_HeizungZone_8_Konfig",
    "ST110_VARExtr_0_HeizungZone_8_Regler_X",
    "ST110_VARExtr_0_HeizungZone_8_Regler_Y",
    "ST110_VARExtr_0_HeizungZone_8_SollTemp",
    "ST110_VARExtr_0_HeizungZone_9_Konfig",
    "ST110_VARExtr_0_HeizungZone_9_Regler_X",
    "ST110_VARExtr_0_HeizungZone_9_Regler_Y",
    "ST110_VARExtr_0_HeizungZone_9_SollTemp",

    # Extruder 1 — Pressure, Heating Zones 1-8, Melt Temp
    "ST110_VARExtr_1_druck_1_IstP",
    "ST110_VARExtr_1_HeizungZone_1_ActEffectPower",
    "ST110_VARExtr_1_HeizungZone_1_Konfig",
    "ST110_VARExtr_1_HeizungZone_1_Regler_X",
    "ST110_VARExtr_1_HeizungZone_1_Regler_Y",
    "ST110_VARExtr_1_HeizungZone_1_SollTemp",
    "ST110_VARExtr_1_HeizungZone_2_ActEffectPower",
    "ST110_VARExtr_1_HeizungZone_2_Konfig",
    "ST110_VARExtr_1_HeizungZone_2_Regler_X",
    "ST110_VARExtr_1_HeizungZone_2_Regler_Y",
    "ST110_VARExtr_1_HeizungZone_2_SollTemp",
    "ST110_VARExtr_1_HeizungZone_3_ActEffectPower",
    "ST110_VARExtr_1_HeizungZone_3_Konfig",
    "ST110_VARExtr_1_HeizungZone_3_Regler_X",
    "ST110_VARExtr_1_HeizungZone_3_Regler_Y",
    "ST110_VARExtr_1_HeizungZone_3_SollTemp",
    "ST110_VARExtr_1_HeizungZone_4_ActEffectPower",
    "ST110_VARExtr_1_HeizungZone_4_Konfig",
    "ST110_VARExtr_1_HeizungZone_4_Regler_X",
    "ST110_VARExtr_1_HeizungZone_4_Regler_Y",
    "ST110_VARExtr_1_HeizungZone_4_SollTemp",
    "ST110_VARExtr_1_HeizungZone_5_ActEffectPower",
    "ST110_VARExtr_1_HeizungZone_5_Konfig",
    "ST110_VARExtr_1_HeizungZone_5_Regler_X",
    "ST110_VARExtr_1_HeizungZone_5_Regler_Y",
    "ST110_VARExtr_1_HeizungZone_5_SollTemp",
    "ST110_VARExtr_1_HeizungZone_6_ActEffectPower",
    "ST110_VARExtr_1_HeizungZone_6_Konfig",
    "ST110_VARExtr_1_HeizungZone_6_Regler_X",
    "ST110_VARExtr_1_HeizungZone_6_Regler_Y",
    "ST110_VARExtr_1_HeizungZone_6_SollTemp",
    "ST110_VARExtr_1_HeizungZone_7_ActEffectPower",
    "ST110_VARExtr_1_HeizungZone_7_Konfig",
    "ST110_VARExtr_1_HeizungZone_7_Regler_X",
    "ST110_VARExtr_1_HeizungZone_7_Regler_Y",
    "ST110_VARExtr_1_HeizungZone_7_SollTemp",
    "ST110_VARExtr_1_HeizungZone_8_ActEffectPower",
    "ST110_VARExtr_1_HeizungZone_8_Konfig",
    "ST110_VARExtr_1_HeizungZone_8_Regler_X",
    "ST110_VARExtr_1_HeizungZone_8_Regler_Y",
    "ST110_VARExtr_1_HeizungZone_8_SollTemp",
    "ST110_VARExtr_1_Massetemperatur",

    # Extruder 2 — Pressure, Heating Zones 1-8, Melt Temp
    "ST110_VARExtr_2_druck_1_IstP",
    "ST110_VARExtr_2_HeizungZone_1_ActEffectPower",
    "ST110_VARExtr_2_HeizungZone_1_Konfig",
    "ST110_VARExtr_2_HeizungZone_1_Regler_X",
    "ST110_VARExtr_2_HeizungZone_1_Regler_Y",
    "ST110_VARExtr_2_HeizungZone_1_SollTemp",
    "ST110_VARExtr_2_HeizungZone_2_ActEffectPower",
    "ST110_VARExtr_2_HeizungZone_2_Konfig",
    "ST110_VARExtr_2_HeizungZone_2_Regler_X",
    "ST110_VARExtr_2_HeizungZone_2_Regler_Y",
    "ST110_VARExtr_2_HeizungZone_2_SollTemp",
    "ST110_VARExtr_2_HeizungZone_3_ActEffectPower",
    "ST110_VARExtr_2_HeizungZone_3_Konfig",
    "ST110_VARExtr_2_HeizungZone_3_Regler_X",
    "ST110_VARExtr_2_HeizungZone_3_Regler_Y",
    "ST110_VARExtr_2_HeizungZone_3_SollTemp",
    "ST110_VARExtr_2_HeizungZone_4_ActEffectPower",
    "ST110_VARExtr_2_HeizungZone_4_Konfig",
    "ST110_VARExtr_2_HeizungZone_4_Regler_X",
    "ST110_VARExtr_2_HeizungZone_4_Regler_Y",
    "ST110_VARExtr_2_HeizungZone_4_SollTemp",
    "ST110_VARExtr_2_HeizungZone_5_ActEffectPower",
    "ST110_VARExtr_2_HeizungZone_5_Konfig",
    "ST110_VARExtr_2_HeizungZone_5_Regler_X",
    "ST110_VARExtr_2_HeizungZone_5_Regler_Y",
    "ST110_VARExtr_2_HeizungZone_5_SollTemp",
    "ST110_VARExtr_2_HeizungZone_6_ActEffectPower",
    "ST110_VARExtr_2_HeizungZone_6_Konfig",
    "ST110_VARExtr_2_HeizungZone_6_Regler_X",
    "ST110_VARExtr_2_HeizungZone_6_Regler_Y",
    "ST110_VARExtr_2_HeizungZone_6_SollTemp",
    "ST110_VARExtr_2_HeizungZone_7_ActEffectPower",
    "ST110_VARExtr_2_HeizungZone_7_Konfig",
    "ST110_VARExtr_2_HeizungZone_7_Regler_X",
    "ST110_VARExtr_2_HeizungZone_7_Regler_Y",
    "ST110_VARExtr_2_HeizungZone_7_SollTemp",
    "ST110_VARExtr_2_HeizungZone_8_ActEffectPower",
    "ST110_VARExtr_2_HeizungZone_8_Konfig",
    "ST110_VARExtr_2_HeizungZone_8_Regler_X",
    "ST110_VARExtr_2_HeizungZone_8_Regler_Y",
    "ST110_VARExtr_2_HeizungZone_8_SollTemp",
    "ST110_VARExtr_2_Massetemperatur",

    # Extruder 3 — Pressure, Heating Zones 1-8, Melt Temp
    "ST110_VARExtr_3_druck_1_IstP",
    "ST110_VARExtr_3_HeizungZone_1_ActEffectPower",
    "ST110_VARExtr_3_HeizungZone_1_Konfig",
    "ST110_VARExtr_3_HeizungZone_1_Regler_X",
    "ST110_VARExtr_3_HeizungZone_1_Regler_Y",
    "ST110_VARExtr_3_HeizungZone_1_SollTemp",
    "ST110_VARExtr_3_HeizungZone_2_ActEffectPower",
    "ST110_VARExtr_3_HeizungZone_2_Konfig",
    "ST110_VARExtr_3_HeizungZone_2_Regler_X",
    "ST110_VARExtr_3_HeizungZone_2_Regler_Y",
    "ST110_VARExtr_3_HeizungZone_2_SollTemp",
    "ST110_VARExtr_3_HeizungZone_3_ActEffectPower",
    "ST110_VARExtr_3_HeizungZone_3_Konfig",
    "ST110_VARExtr_3_HeizungZone_3_Regler_X",
    "ST110_VARExtr_3_HeizungZone_3_Regler_Y",
    "ST110_VARExtr_3_HeizungZone_3_SollTemp",
    "ST110_VARExtr_3_HeizungZone_4_ActEffectPower",
    "ST110_VARExtr_3_HeizungZone_4_Konfig",
    "ST110_VARExtr_3_HeizungZone_4_Regler_X",
    "ST110_VARExtr_3_HeizungZone_4_Regler_Y",
    "ST110_VARExtr_3_HeizungZone_4_SollTemp",
    "ST110_VARExtr_3_HeizungZone_5_ActEffectPower",
    "ST110_VARExtr_3_HeizungZone_5_Konfig",
    "ST110_VARExtr_3_HeizungZone_5_Regler_X",
    "ST110_VARExtr_3_HeizungZone_5_Regler_Y",
    "ST110_VARExtr_3_HeizungZone_5_SollTemp",
    "ST110_VARExtr_3_HeizungZone_6_ActEffectPower",
    "ST110_VARExtr_3_HeizungZone_6_Konfig",
    "ST110_VARExtr_3_HeizungZone_6_Regler_X",
    "ST110_VARExtr_3_HeizungZone_6_Regler_Y",
    "ST110_VARExtr_3_HeizungZone_6_SollTemp",
    "ST110_VARExtr_3_HeizungZone_7_ActEffectPower",
    "ST110_VARExtr_3_HeizungZone_7_Konfig",
    "ST110_VARExtr_3_HeizungZone_7_Regler_X",
    "ST110_VARExtr_3_HeizungZone_7_Regler_Y",
    "ST110_VARExtr_3_HeizungZone_7_SollTemp",
    "ST110_VARExtr_3_HeizungZone_8_ActEffectPower",
    "ST110_VARExtr_3_HeizungZone_8_Konfig",
    "ST110_VARExtr_3_HeizungZone_8_Regler_X",
    "ST110_VARExtr_3_HeizungZone_8_Regler_Y",
    "ST110_VARExtr_3_HeizungZone_8_SollTemp",
    "ST110_VARExtr_3_Massetemperatur",

    # Extrusion general — Ex-level variables
    "ST110_VAREx_0_Dos_0_IstLMGewicht",
    "ST110_VAREx_0_GesamtDS",
    "ST110_VAREx_0_MischDicht",
    "ST110_VAREx_0_RegelungEin",
    "ST110_VAREx_0_SDickeIst",
    "ST110_VAREx_0_SDickeProz",
    "ST110_VAREx_0_SDickeSoll",
    "ST110_VAREx_0_SollDS",
    "ST110_VAREx_0_SollLM",

    # Extruder 1 dosing (Dos 0-5)
    "ST110_VAREx_1_DissipationPwr",
    "ST110_VAREx_1_Dos_0_IstLMGewicht",
    "ST110_VAREx_1_Dos_0_Ventil",
    "ST110_VAREx_1_Dos_1_IstForederrate",
    "ST110_VAREx_1_Dos_2_IstAnteil",
    "ST110_VAREx_1_Dos_2_IstDrehzahl",
    "ST110_VAREx_1_Dos_2_IstDurchsatz",
    "ST110_VAREx_1_Dos_2_IstForederrate",
    "ST110_VAREx_1_Dos_2_Materialvorwahl",
    "ST110_VAREx_1_Dos_2_SchuettDichte",
    "ST110_VAREx_1_Dos_2_SollAnteil",
    "ST110_VAREx_1_Dos_2_SollDichte",
    "ST110_VAREx_1_Dos_2_SollStatus",
    "ST110_VAREx_1_Dos_2_Ventil",
    "ST110_VAREx_1_Dos_3_IstAnteil",
    "ST110_VAREx_1_Dos_3_IstDrehzahl",
    "ST110_VAREx_1_Dos_3_IstDurchsatz",
    "ST110_VAREx_1_Dos_3_IstForederrate",
    "ST110_VAREx_1_Dos_3_Materialvorwahl",
    "ST110_VAREx_1_Dos_3_SchuettDichte",
    "ST110_VAREx_1_Dos_3_SollAnteil",
    "ST110_VAREx_1_Dos_3_SollDichte",
    "ST110_VAREx_1_Dos_3_SollStatus",
    "ST110_VAREx_1_Dos_3_Ventil",
    "ST110_VAREx_1_Dos_4_IstAnteil",
    "ST110_VAREx_1_Dos_4_IstDrehzahl",
    "ST110_VAREx_1_Dos_4_IstDurchsatz",
    "ST110_VAREx_1_Dos_4_IstForederrate",
    "ST110_VAREx_1_Dos_4_Materialvorwahl",
    "ST110_VAREx_1_Dos_4_SchuettDichte",
    "ST110_VAREx_1_Dos_4_SollAnteil",
    "ST110_VAREx_1_Dos_4_SollDichte",
    "ST110_VAREx_1_Dos_4_SollStatus",
    "ST110_VAREx_1_Dos_4_Ventil",
    "ST110_VAREx_1_Dos_5_IstAnteil",
    "ST110_VAREx_1_Dos_5_IstDrehzahl",
    "ST110_VAREx_1_Dos_5_IstDurchsatz",
    "ST110_VAREx_1_Dos_5_IstForederrate",
    "ST110_VAREx_1_Dos_5_Materialvorwahl",
    "ST110_VAREx_1_Dos_5_SchuettDichte",
    "ST110_VAREx_1_Dos_5_SollAnteil",
    "ST110_VAREx_1_Dos_5_SollDichte",
    "ST110_VAREx_1_Dos_5_SollStatus",
    "ST110_VAREx_1_Dos_5_Ventil",
    "ST110_VAREx_1_Foerderrate",
    "ST110_VAREx_1_GesamtDS",
    "ST110_VAREx_1_MischDicht",
    "ST110_VAREx_1_MischerMotor",
    "ST110_VAREx_1_RegelungEin",
    "ST110_VAREx_1_SDickeIst",
    "ST110_VAREx_1_SDickeProz",
    "ST110_VAREx_1_SDickeSoll",
    "ST110_VAREx_1_SollLM",

    # Extruder 2 dosing
    "ST110_VAREx_2_DissipationPwr",
    "ST110_VAREx_2_Dos_0_IstLMGewicht",
    "ST110_VAREx_2_Dos_0_Ventil",
    "ST110_VAREx_2_Dos_1_IstForederrate",
    "ST110_VAREx_2_Dos_2_IstAnteil",
    "ST110_VAREx_2_Dos_2_IstDrehzahl",
    "ST110_VAREx_2_Dos_2_IstDurchsatz",
    "ST110_VAREx_2_Dos_2_IstForederrate",
    "ST110_VAREx_2_Dos_2_Materialvorwahl",
    "ST110_VAREx_2_Dos_2_SchuettDichte",
    "ST110_VAREx_2_Dos_2_SollAnteil",
    "ST110_VAREx_2_Dos_2_SollDichte",
    "ST110_VAREx_2_Dos_2_SollStatus",
    "ST110_VAREx_2_Dos_2_Ventil",
    "ST110_VAREx_2_Dos_3_IstAnteil",
    "ST110_VAREx_2_Dos_3_IstDrehzahl",
    "ST110_VAREx_2_Dos_3_IstDurchsatz",
    "ST110_VAREx_2_Dos_3_IstForederrate",
    "ST110_VAREx_2_Dos_3_Materialvorwahl",
    "ST110_VAREx_2_Dos_3_SchuettDichte",
    "ST110_VAREx_2_Dos_3_SollAnteil",
    "ST110_VAREx_2_Dos_3_SollDichte",
    "ST110_VAREx_2_Dos_3_SollStatus",
    "ST110_VAREx_2_Dos_3_Ventil",
    "ST110_VAREx_2_Dos_4_IstAnteil",
    "ST110_VAREx_2_Dos_4_IstDrehzahl",
    "ST110_VAREx_2_Dos_4_IstDurchsatz",
    "ST110_VAREx_2_Dos_4_IstForederrate",
    "ST110_VAREx_2_Dos_4_Materialvorwahl",
    "ST110_VAREx_2_Dos_4_SchuettDichte",
    "ST110_VAREx_2_Dos_4_SollAnteil",
    "ST110_VAREx_2_Dos_4_SollDichte",
    "ST110_VAREx_2_Dos_4_SollStatus",
    "ST110_VAREx_2_Dos_4_Ventil",
    "ST110_VAREx_2_Dos_5_IstAnteil",
    "ST110_VAREx_2_Dos_5_IstDrehzahl",
    "ST110_VAREx_2_Dos_5_IstDurchsatz",
    "ST110_VAREx_2_Dos_5_IstForederrate",
    "ST110_VAREx_2_Dos_5_Materialvorwahl",
    "ST110_VAREx_2_Dos_5_SchuettDichte",
    "ST110_VAREx_2_Dos_5_SollAnteil",
    "ST110_VAREx_2_Dos_5_SollDichte",
    "ST110_VAREx_2_Dos_5_SollStatus",
    "ST110_VAREx_2_Dos_5_Ventil",
    "ST110_VAREx_2_Foerderrate",
    "ST110_VAREx_2_GesamtDS",
    "ST110_VAREx_2_MischDicht",
    "ST110_VAREx_2_MischerMotor",
    "ST110_VAREx_2_RegelungEin",
    "ST110_VAREx_2_SDickeIst",
    "ST110_VAREx_2_SDickeProz",
    "ST110_VAREx_2_SDickeSoll",
    "ST110_VAREx_2_SollLM",

    # Extruder 3 dosing
    "ST110_VAREx_3_DissipationPwr",
    "ST110_VAREx_3_Dos_0_IstLMGewicht",
    "ST110_VAREx_3_Dos_0_Ventil",
    "ST110_VAREx_3_Dos_1_IstForederrate",
    "ST110_VAREx_3_Dos_2_IstAnteil",
    "ST110_VAREx_3_Dos_2_IstDrehzahl",
    "ST110_VAREx_3_Dos_2_IstDurchsatz",
    "ST110_VAREx_3_Dos_2_IstForederrate",
    "ST110_VAREx_3_Dos_2_Materialvorwahl",
    "ST110_VAREx_3_Dos_2_SchuettDichte",
    "ST110_VAREx_3_Dos_2_SollAnteil",
    "ST110_VAREx_3_Dos_2_SollDichte",
    "ST110_VAREx_3_Dos_2_SollStatus",
    "ST110_VAREx_3_Dos_2_Ventil",
    "ST110_VAREx_3_Dos_3_IstAnteil",
    "ST110_VAREx_3_Dos_3_IstDrehzahl",
    "ST110_VAREx_3_Dos_3_IstDurchsatz",
    "ST110_VAREx_3_Dos_3_IstForederrate",
    "ST110_VAREx_3_Dos_3_Materialvorwahl",
    "ST110_VAREx_3_Dos_3_SchuettDichte",
    "ST110_VAREx_3_Dos_3_SollAnteil",
    "ST110_VAREx_3_Dos_3_SollDichte",
    "ST110_VAREx_3_Dos_3_SollStatus",
    "ST110_VAREx_3_Dos_3_Ventil",
    "ST110_VAREx_3_Dos_4_IstAnteil",
    "ST110_VAREx_3_Dos_4_IstDrehzahl",
    "ST110_VAREx_3_Dos_4_IstDurchsatz",
    "ST110_VAREx_3_Dos_4_IstForederrate",
    "ST110_VAREx_3_Dos_4_Materialvorwahl",
    "ST110_VAREx_3_Dos_4_SchuettDichte",
    "ST110_VAREx_3_Dos_4_SollAnteil",
    "ST110_VAREx_3_Dos_4_SollDichte",
    "ST110_VAREx_3_Dos_4_SollStatus",
    "ST110_VAREx_3_Dos_4_Ventil",
    "ST110_VAREx_3_Dos_5_IstAnteil",
    "ST110_VAREx_3_Dos_5_IstDrehzahl",
    "ST110_VAREx_3_Dos_5_IstDurchsatz",
    "ST110_VAREx_3_Dos_5_IstForederrate",
    "ST110_VAREx_3_Dos_5_Materialvorwahl",
    "ST110_VAREx_3_Dos_5_SchuettDichte",
    "ST110_VAREx_3_Dos_5_SollAnteil",
    "ST110_VAREx_3_Dos_5_SollDichte",
    "ST110_VAREx_3_Dos_5_SollStatus",
    "ST110_VAREx_3_Dos_5_Ventil",
    "ST110_VAREx_3_Foerderrate",
    "ST110_VAREx_3_GesamtDS",
    "ST110_VAREx_3_MischDicht",
    "ST110_VAREx_3_MischerMotor",
    "ST110_VAREx_3_RegelungEin",
    "ST110_VAREx_3_SDickeIst",
    "ST110_VAREx_3_SDickeProz",
    "ST110_VAREx_3_SDickeSoll",
    "ST110_VAREx_3_SollLM",

    # Blowers
    "ST110_VARGeblaese_1_Auslastung",
    "ST110_VARGeblaese_2_Auslastung",
    "ST110_VARGeblaese_3_Auslastung",

    # IBC
    "ST110_VARIBC_1_Ist_n_Calc",
    "ST110_VARIBC_1_Soll_n_Visu",
    "ST110_VARIBC_2_Ist_n_Calc",
    "ST110_VARIBC_2_Soll_n_Visu",
    "ST110_VARIBC_3_Ist_n_Calc",
    "ST110_VARIBC_3_Soll_n_Visu",

    # Basket
    "ST110_VARKorbBreiteIstBreite",
    "ST110_VARKorb_dIst_d",
    "ST110_VARKorb_dSoll_d",

    # Cooling devices
    "ST110_VARKuehlGeraet_1_IstTemp",
    "ST110_VARKuehlGeraet_1_SollTemp",
    "ST110_VARKuehlGeraet_2_IstTemp",
    "ST110_VARKuehlGeraet_2_SollTemp",
    "ST110_VARKuehlGeraet_3_IstTemp",
    "ST110_VARKuehlGeraet_3_SollTemp",

    # Profile commands
    "ST110_VARProfilCmdBefehl",
    "ST110_VARProfilCmdMessung",
    "ST110_VARProfilCmdReglerEin",
    "ST110_VARProfilCmdReglerStop",

    # Hoppers
    "ST110_VARTrichter_10_Vorwahl",
    "ST110_VARTrichter_11_Vorwahl",
    "ST110_VARTrichter_15_Vorwahl",
    "ST110_VARTrichter_16_Vorwahl",
    "ST110_VARTrichter_17_Vorwahl",
    "ST110_VARTrichter_18_Vorwahl",
    "ST110_VARTrichter_1_Vorwahl",
    "ST110_VARTrichter_2_Vorwahl",
    "ST110_VARTrichter_3_Vorwahl",
    "ST110_VARTrichter_4_Vorwahl",
    "ST110_VARTrichter_8_Vorwahl",
    "ST110_VARTrichter_9_Vorwahl",

    # USB regulation
    "ST110_VARUSBRegelungIstwert",
    "ST110_VARUSBRegelungPosKlappe",
    "ST110_VARUSBRegelungRegelungEin",
    "ST110_VARUSBRegelungSollwert",

    # Haul-off (ST112)
    "ST112_VARAbzug_1_IstEin",
    "ST112_VARAbzug_1_IstZu",
    "ST112_VARAbzug_1_SollSpeed",
    "ST112_VARHoursCntAbzugEin",
    "ST112_VARHoursCntAbzugWartung",
    "ST112_VARReversierungSollEin",
    "ST112_VARReversierungSollSpeed",
    "ST112_VARReversierungSollWinkel",
    "ST112_VARReversierungZeit",
    "ST112_VARSeitenfuehrungUntenAufVisu",
    "ST112_VARSeitenfuehrungUntenZuVisu",
    "ST112_VARWendestangenSollwert180O",
    "ST112_VARWendestangenSollwert180U",
    "ST112_VARWendestangenSollwert360O",
    "ST112_VARWendestangenSollwert360U",

    # Winder 1 (ST113)
    "ST113_VARActLen",
    "ST113_VARCdEnable",
    "ST113_VARCdSpeedHMI",
    "ST113_VARCdSpSpeed",
    "ST113_VARCdSpTens",
    "ST113_VARCdTensVis",
    "ST113_VARClpEnable",
    "ST113_VARClpNum",
    "ST113_VARClpReductVal",
    "ST113_VARClpTens",
    "ST113_VARCntStart",
    "ST113_VARDiaNextRollRc",
    "ST113_VARDiaRollPanel",
    "ST113_VARDiaRollRc",
    "ST113_VARDiaShaft",
    "ST113_VARGapDiff",
    "ST113_VARGapEnable",
    "ST113_VARGapMax",
    "ST113_VARGapMin",
    "ST113_VARGesamtZeit",
    "ST113_VARInitLayOnPress",
    "ST113_VARLmpRun",
    "ST113_VARPivotLoadCmd",
    "ST113_VARPnCloseCmd",
    "ST113_VARPnEdgeCmd",
    "ST113_VARPnSpeedHMI",
    "ST113_VARPnSpSpeed",
    "ST113_VARPnSpTens",
    "ST113_VARPnTensVis",
    "ST113_VARPrevLen",
    "ST113_VARRemainingLen",
    "ST113_VARRemainingTimeVis",
    "ST113_VARRotationReverse",
    "ST113_VARSwdSpeedHMI",
    "ST113_VARSwdSpSpeed",
    "ST113_VARTagZeit",
    "ST113_VARTapeNum",
    "ST113_VARTapeReductVal",
    "ST113_VARTargetLenActRoll",
    "ST113_VARTargetLenNextRoll",
    "ST113_VARTensPlusEnable",
    "ST113_VARTensPlusSp",
    "ST113_VARTotalLen",
    "ST113_VARTotalRolls",
    "ST113_VARWdCloseCmd",
    "ST113_VARWdSpeedHMI",
    "ST113_VARWdSpSpeed",
    "ST113_VARWdSpTens",
    "ST113_VARWdTapeNum",
    "ST113_VARWdTapeReductVal",
    "ST113_VARWdTensVis",

    # Winder 2 (ST114)
    "ST114_VARActLen",
    "ST114_VARCdEnable",
    "ST114_VARCdSpeedHMI",
    "ST114_VARCdSpSpeed",
    "ST114_VARCdSpTens",
    "ST114_VARCdTensVis",
    "ST114_VARClpEnable",
    "ST114_VARClpNum",
    "ST114_VARClpReductVal",
    "ST114_VARClpTens",
    "ST114_VARCntStart",
    "ST114_VARDiaNextRollRc",
    "ST114_VARDiaRollPanel",
    "ST114_VARDiaRollRc",
    "ST114_VARDiaShaft",
    "ST114_VARGapDiff",
    "ST114_VARGapEnable",
    "ST114_VARGapMax",
    "ST114_VARGapMin",
    "ST114_VARGesamtZeit",
    "ST114_VARInitLayOnPress",
    "ST114_VARLmpRun",
    "ST114_VARPrevLen",
    "ST114_VARRemainingLen",
    "ST114_VARRemainingTimeVis",
    "ST114_VARRotationReverse",
    "ST114_VARSwdSpeedHMI",
    "ST114_VARSwdSpSpeed",
    "ST114_VARTagZeit",
    "ST114_VARTapeNum",
    "ST114_VARTapeReductVal",
    "ST114_VARTargetLenActRoll",
    "ST114_VARTargetLenNextRoll",
    "ST114_VARTensPlusEnable",
    "ST114_VARTensPlusSp",
    "ST114_VARTotalLen",
    "ST114_VARTotalRolls",
    "ST114_VARWdCloseCmd",
    "ST114_VARWdSpeedHMI",
    "ST114_VARWdSpSpeed",
    "ST114_VARWdSpTens",
    "ST114_VARWdTapeNum",
    "ST114_VARWdTapeReductVal",
    "ST114_VARWdTensVis",
]

OUTPUT_COLS: List[str] = [
    "ST110_VAREx_1_SDickeIst",
    "ST110_VAREx_2_SDickeIst",
    "ST110_VAREx_3_SDickeIst",
    "ST110_VARExtr_1_Massetemperatur",
    "ST110_VARExtr_2_Massetemperatur",
    "ST110_VARExtr_3_Massetemperatur",
    "ST110_VARExtr_1_druck_1_IstP",
    "ST110_VARExtr_2_druck_1_IstP",
    "ST110_VARExtr_3_druck_1_IstP",
    "ST110_VARIBC_1_Ist_n_Calc",
    "ST110_VARIBC_2_Ist_n_Calc",
    "ST110_VARIBC_3_Ist_n_Calc",
    "ST110_VARKuehlGeraet_1_IstTemp",
    "ST110_VARKuehlGeraet_2_IstTemp",
    "ST110_VARKuehlGeraet_3_IstTemp",
    "ST112_VARAbzug_1_IstZu",
    "ST113_VARDiaRollRc",
    "ST113_VARRemainingLen",
    "ST113_VARWdSpTens",
    "ST114_VARDiaRollRc",
    "ST114_VARRemainingLen",
    "ST114_VARWdSpTens",
]

INPUT_COLS: List[str] = [
    "ST110_VARExtr_1_HeizungZone_1_SollTemp",
    "ST110_VARExtr_1_HeizungZone_2_SollTemp",
    "ST110_VARExtr_1_HeizungZone_3_SollTemp",
    "ST110_VARExtr_1_HeizungZone_4_SollTemp",
    "ST110_VARExtr_2_HeizungZone_1_SollTemp",
    "ST110_VARExtr_2_HeizungZone_2_SollTemp",
    "ST110_VARExtr_2_HeizungZone_3_SollTemp",
    "ST110_VARExtr_2_HeizungZone_4_SollTemp",
    "ST110_VARExtr_3_HeizungZone_1_SollTemp",
    "ST110_VARExtr_3_HeizungZone_2_SollTemp",
    "ST110_VARExtr_3_HeizungZone_3_SollTemp",
    "ST110_VARExtr_3_HeizungZone_4_SollTemp",
    "ST110_VAREx_1_SDickeSoll",
    "ST110_VAREx_2_SDickeSoll",
    "ST110_VAREx_3_SDickeSoll",
    "ST110_VARIBC_1_Soll_n_Visu",
    "ST110_VARIBC_2_Soll_n_Visu",
    "ST110_VARIBC_3_Soll_n_Visu",
    "ST110_VARKuehlGeraet_1_SollTemp",
    "ST110_VARKuehlGeraet_2_SollTemp",
    "ST110_VARKuehlGeraet_3_SollTemp",
    "ST112_VARAbzug_1_SollSpeed",
    "ST113_VARWdSpSpeed",
    "ST113_VARWdSpTens",
    "ST114_VARWdSpSpeed",
    "ST114_VARWdSpTens",
]


# ---------------------------------------------------------------------------
# Typed configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Parameters governing data loading and preprocessing."""

    sampling_time: float = 3.0          # seconds
    train_fraction: float = 0.70
    outlier_zscore: float = 5.0
    ffill_limit: int = 5
    bfill_limit: int = 5
    time_column: str = "Datum"
    synthetic_samples: int = 3_000
    synthetic_noise_std: float = 0.02
    random_seed: int = 42


@dataclass(frozen=True)
class IdentificationConfig:
    """Parameters for N4SID subspace identification."""

    n_states: int = 20
    n_block_rows: int = 30
    optimise_params: bool = True
    optimisation_method: str = "L-BFGS-B"
    optimisation_max_iter: int = 100
    optimisation_samples: int = 500


@dataclass(frozen=True)
class ReductionConfig:
    """Parameters for model order reduction."""

    n_states_bt: int = 12               # target order after balanced truncation
    bt_energy_tolerance: float = 0.99   # fraction of HSV energy to retain
    pod_energy_tolerance: float = 0.99  # fraction of POD energy to retain


@dataclass(frozen=True)
class KalmanConfig:
    """Kalman filter noise covariance parameters."""

    process_noise_scale: float = 1e-3
    measurement_noise_scale: float = 1e-2


@dataclass(frozen=True)
class MPCConfig:
    """MPC horizon, constraint and weight parameters."""

    prediction_horizon: int = 20
    control_horizon: int = 8
    u_bound: float = 3.0                # ±σ in normalised space
    du_bound: float = 0.5
    y_bound: float = 5.0
    q_thickness_weight: float = 10.0    # layer thickness priority
    q_temperature_weight: float = 5.0   # melt temperature priority
    q_default_weight: float = 1.0
    r_weight: float = 0.05
    s_weight: float = 0.01
    optimise_weights: bool = True
    weight_opt_iterations: int = 20
    weight_opt_lambda: float = 0.1


@dataclass(frozen=True)
class SimulationConfig:
    """Closed-loop simulation parameters."""

    n_steps: int = 300
    noise_std: float = 0.02
    ref_step_time_1: int = 50
    ref_step_time_2: int = 150
    ref_step_time_3: int = 250
    ref_amplitude_1: float = 0.5
    ref_amplitude_2: float = -0.3


@dataclass
class ProjectConfig:
    """
    Top-level project configuration aggregating all sub-configs.

    Usage
    -----
    >>> cfg = ProjectConfig()
    >>> cfg.data.sampling_time
    3.0
    """

    data: DataConfig = field(default_factory=DataConfig)
    identification: IdentificationConfig = field(
        default_factory=IdentificationConfig
    )
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    output_dir: str = "outputs"
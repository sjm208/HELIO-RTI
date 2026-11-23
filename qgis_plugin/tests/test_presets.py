"""The Presets menu, and the coloured lights with a colour composite."""
import numpy as np

from helio_relief import dialog as D, helio, relief_engine as eng


def test_every_object_preset_names_a_real_look_and_surface():
    for name, (look, surface) in D.OBJECT_PRESETS.items():
        assert look in D.LOOKS, name
        assert surface in D.SURFACE_PRESETS, name


def test_helio_object_types_have_a_preset():
    assert set(D.HELIO_OBJECTS.values()) <= set(D.OBJECT_PRESETS)


def test_the_gallery_lists_only_real_looks():
    assert set(D.STRIP_LOOKS) <= set(D.LOOKS)


def test_coloured_lights_take_a_colour_composite():
    shade = np.full((6, 8, 3), 0.5, np.float32)
    albedo = np.full((6, 8, 3), 0.8, np.float32)
    albedo[0, 0] = 0
    out = eng.apply_albedo(shade, albedo)
    assert out.shape == (6, 8, 3) and (out[0, 0] == 0.5).all()


def test_grey_shading_still_does():
    out = eng.apply_albedo(np.full((6, 8), 0.5, np.float32), np.full((6, 8, 3), 0.8, np.float32))
    assert out.shape == (6, 8, 3)


def test_the_readme_says_the_object_type():
    assert helio.parse_readme("Object ID     : X\nObject type   : seal\n")["object_type"] == "seal"

from cache_manager import clear_screenshot_cache


def test_clear_screenshot_cache_keeps_template_sources(tmp_path):
    debug_dir = tmp_path / "debug"
    (debug_dir / "template_sources").mkdir(parents=True)
    (debug_dir / "template_match_checks").mkdir(parents=True)
    generated = debug_dir / "template_match_checks" / "match.png"
    generated.write_bytes(b"generated")
    root_capture = debug_dir / "client_capture.png"
    root_capture.write_bytes(b"capture")
    source = debug_dir / "template_sources" / "keep.png"
    source.write_bytes(b"template source")

    result = clear_screenshot_cache(debug_dir)

    assert result.removed_files == 2
    assert not generated.exists()
    assert not root_capture.exists()
    assert source.exists()

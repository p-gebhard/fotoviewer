from PIL import Image

from fotoviewer.services import duplicate_cleanup
from fotoviewer.services.duplicate_cleanup import move_to_trash
from fotoviewer.services.image_files import find_image_paths


def make_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color="red").save(path)


def test_move_to_trash_moves_file_into_dated_subfolder(tmp_path):
    make_image(tmp_path / "a.jpg")

    result = move_to_trash(str(tmp_path), ["a.jpg"])

    assert result["moved"] == ["a.jpg"]
    assert result["failed"] == []
    assert not (tmp_path / "a.jpg").exists()
    trash_files = list((tmp_path / ".duplicates").glob("*/a.jpg"))
    assert len(trash_files) == 1


def test_move_to_trash_writes_manifest_with_original_paths(tmp_path):
    make_image(tmp_path / "a.jpg")
    make_image(tmp_path / "Urlaub" / "b.jpg")

    move_to_trash(str(tmp_path), ["a.jpg", "Urlaub/b.jpg"])

    manifest_files = list((tmp_path / ".duplicates").glob("*/manifest.txt"))
    assert len(manifest_files) == 1
    content = manifest_files[0].read_text()
    assert "a.jpg" in content
    assert "Urlaub/b.jpg" in content


def test_move_to_trash_skips_manifest_when_nothing_moved(tmp_path):
    move_to_trash(str(tmp_path), ["does-not-exist.jpg"])

    assert not (tmp_path / ".duplicates").exists()


def test_move_to_trash_preserves_subfolder_structure(tmp_path):
    make_image(tmp_path / "Urlaub" / "a.jpg")

    move_to_trash(str(tmp_path), ["Urlaub/a.jpg"])

    trash_files = list((tmp_path / ".duplicates").glob("*/Urlaub/a.jpg"))
    assert len(trash_files) == 1


def test_move_to_trash_reports_missing_file_as_failed_without_aborting(tmp_path):
    make_image(tmp_path / "a.jpg")

    result = move_to_trash(str(tmp_path), ["does-not-exist.jpg", "a.jpg"])

    assert result["failed"] == ["does-not-exist.jpg"]
    assert result["moved"] == ["a.jpg"]


def test_move_to_trash_blocks_path_traversal(tmp_path):
    outside = tmp_path.parent / "outside_secret.jpg"
    make_image(outside)

    result = move_to_trash(str(tmp_path), ["../outside_secret.jpg"])

    assert result["failed"] == ["../outside_secret.jpg"]
    assert result["moved"] == []
    assert outside.exists()


def test_move_to_trash_reports_move_failure_without_aborting(tmp_path, monkeypatch):
    make_image(tmp_path / "a.jpg")
    make_image(tmp_path / "b.jpg")
    original_move = duplicate_cleanup.shutil.move

    def failing_move(src, dst):
        if src.endswith("a.jpg"):
            raise OSError("simulated failure")
        return original_move(src, dst)

    monkeypatch.setattr(duplicate_cleanup.shutil, "move", failing_move)

    result = move_to_trash(str(tmp_path), ["a.jpg", "b.jpg"])

    assert result["failed"] == ["a.jpg"]
    assert result["moved"] == ["b.jpg"]


def test_duplicates_folder_is_excluded_from_find_image_paths(tmp_path):
    make_image(tmp_path / "a.jpg")
    move_to_trash(str(tmp_path), ["a.jpg"])

    paths = find_image_paths(str(tmp_path))

    assert paths == []

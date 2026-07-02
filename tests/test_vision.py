import numpy as np

from vision import (
    Point,
    Rect,
    RecognitionSize,
    crop_roi,
    map_point_to_screen,
    map_rect_to_screen,
    match_template,
    resize_to_recognition,
)


def test_resize_to_recognition_uses_configured_size():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    resized = resize_to_recognition(image, RecognitionSize(width=1280, height=720))

    assert resized.shape == (720, 1280, 3)


def test_map_point_and_rect_from_recognition_to_screen_coordinates():
    recognition = RecognitionSize(width=1280, height=720)
    actual = RecognitionSize(width=2560, height=1440)

    assert map_point_to_screen(Point(x=100, y=50), recognition, actual) == Point(x=200, y=100)
    assert map_rect_to_screen(Rect(x=10, y=20, width=30, height=40), recognition, actual) == Rect(
        x=20,
        y=40,
        width=60,
        height=80,
    )


def test_crop_roi_clips_to_image_bounds():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    cropped, clipped = crop_roi(image, Rect(x=150, y=80, width=100, height=50))

    assert clipped == Rect(x=150, y=80, width=50, height=20)
    assert cropped.shape == (20, 50, 3)


def test_match_template_returns_highest_score_center_and_rect_with_roi():
    recognition = RecognitionSize(width=1280, height=720)
    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    template = np.zeros((30, 40, 3), dtype=np.uint8)
    template[5:25, 10:30] = (255, 255, 255)
    template[12:18, 16:24] = (0, 0, 0)
    screen[205:235, 310:350] = template

    result = match_template(
        screen,
        template,
        threshold=0.95,
        roi=Rect(x=280, y=180, width=120, height=100),
        recognition_size=recognition,
        actual_size=recognition,
    )

    assert result.found is True
    assert result.score > 0.99
    assert result.rect == Rect(x=310, y=205, width=40, height=30)
    assert result.center == Point(x=330, y=220)
    assert result.screen_center == Point(x=330, y=220)

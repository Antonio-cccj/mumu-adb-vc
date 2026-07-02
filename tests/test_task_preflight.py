from pathlib import Path

from task_engine import TaskDefinition
from task_preflight import (
    MissingTemplateGroup,
    format_missing_template_groups,
    missing_template_groups,
    missing_template_paths,
)


def test_missing_template_paths_reports_only_unreadable_templates(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "existing.png").write_bytes(b"not a real image but present")
    task = TaskDefinition.model_validate(
        {
            "start": "node",
            "nodes": [
                {
                    "name": "node",
                    "template": "existing.png",
                    "templates": ["missing.png", str(tmp_path / "absolute_missing.png")],
                    "action": "click",
                }
            ],
        }
    )

    missing = missing_template_paths(task, template_dir)

    assert missing == [template_dir / "missing.png", tmp_path / "absolute_missing.png"]


def test_missing_template_groups_require_only_one_candidate_per_node(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "existing.png").write_bytes(b"present")
    task = TaskDefinition.model_validate(
        {
            "start": "start",
            "nodes": [
                {
                    "name": "start",
                    "templates": ["missing_a.png", "existing.png"],
                    "action": "click",
                    "next": "need_one",
                },
                {
                    "name": "need_one",
                    "templates": ["missing_b.png", "missing_c.png"],
                    "action": "click",
                },
            ],
        }
    )

    groups = missing_template_groups(task, template_dir)

    assert groups == [
        MissingTemplateGroup(
            node="need_one",
            candidates=[template_dir / "missing_b.png", template_dir / "missing_c.png"],
            required_count=1,
            existing_count=0,
        )
    ]


def test_missing_template_groups_respects_min_matches(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "existing.png").write_bytes(b"present")
    task = TaskDefinition.model_validate(
        {
            "start": "farm_loaded",
            "nodes": [
                {
                    "name": "farm_loaded",
                    "templates": ["existing.png", "missing_a.png", "missing_b.png"],
                    "min_matches": 2,
                    "action": "wait",
                },
            ],
        }
    )

    groups = missing_template_groups(task, template_dir)

    assert groups == [
        MissingTemplateGroup(
            node="farm_loaded",
            candidates=[template_dir / "existing.png", template_dir / "missing_a.png", template_dir / "missing_b.png"],
            required_count=2,
            existing_count=1,
        )
    ]


def test_missing_template_groups_checks_visual_approach_target_and_success_templates(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "one_key.png").write_bytes(b"present")
    task = TaskDefinition.model_validate(
        {
            "start": "approach",
            "nodes": [
                {
                    "name": "approach",
                    "action": "visual_approach",
                    "approach": {
                        "target_templates": ["missing_statue.png"],
                        "success_templates": ["one_key.png"],
                    },
                },
            ],
        }
    )

    groups = missing_template_groups(task, template_dir)

    assert groups == [
        MissingTemplateGroup(
            node="approach:target",
            candidates=[template_dir / "missing_statue.png"],
            required_count=1,
            existing_count=0,
        )
    ]


def test_missing_template_groups_checks_maturity_anchor_template(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    task = TaskDefinition.model_validate(
        {
            "start": "read_time",
            "nodes": [
                {
                    "name": "read_time",
                    "action": "read_maturity_time",
                    "maturity": {"anchor_template": "missing_shovel.png"},
                }
            ],
        }
    )

    missing_paths = missing_template_paths(task, template_dir)
    groups = missing_template_groups(task, template_dir)

    assert missing_paths == [template_dir / "missing_shovel.png"]
    assert groups == [
        MissingTemplateGroup(
            node="read_time:maturity_anchor",
            candidates=[template_dir / "missing_shovel.png"],
            required_count=1,
            existing_count=0,
        )
    ]


def test_missing_template_groups_checks_maturity_tomorrow_template(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    task = TaskDefinition.model_validate(
        {
            "start": "read_time",
            "nodes": [
                {
                    "name": "read_time",
                    "action": "read_maturity_time",
                    "maturity": {"tomorrow_template": "missing_tomorrow.png"},
                }
            ],
        }
    )

    missing_paths = missing_template_paths(task, template_dir)
    groups = missing_template_groups(task, template_dir)

    assert missing_paths == [template_dir / "missing_tomorrow.png"]
    assert groups == [
        MissingTemplateGroup(
            node="read_time:maturity_tomorrow",
            candidates=[template_dir / "missing_tomorrow.png"],
            required_count=1,
            existing_count=0,
        )
    ]


def test_missing_template_groups_checks_maturity_farm_guard_templates(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "existing_farm.png").write_bytes(b"present")
    task = TaskDefinition.model_validate(
        {
            "start": "read_time",
            "nodes": [
                {
                    "name": "read_time",
                    "action": "read_maturity_time",
                    "maturity": {
                        "farm_templates": ["existing_farm.png", "missing_farm.png"],
                        "farm_min_matches": 2,
                    },
                }
            ],
        }
    )

    missing_paths = missing_template_paths(task, template_dir)
    groups = missing_template_groups(task, template_dir)

    assert missing_paths == [template_dir / "missing_farm.png"]
    assert groups == [
        MissingTemplateGroup(
            node="read_time:maturity_farm",
            candidates=[template_dir / "existing_farm.png", template_dir / "missing_farm.png"],
            required_count=2,
            existing_count=1,
        )
    ]


def test_missing_template_groups_checks_route_navigate_assets(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    route_path = tmp_path / "route.yaml"
    route_path.write_text(
        """
name: demo
success:
  templates: [missing_success.png]
reset:
  templates: [missing_reset.png]
keyframes:
  - name: spawn
    image: missing_spawn.png
    move: left
""",
        encoding="utf-8",
    )
    task = TaskDefinition.model_validate(
        {
            "start": "navigate",
            "nodes": [
                {
                    "name": "navigate",
                    "action": "route_navigate",
                    "route": {"path": str(route_path)},
                }
            ],
        }
    )

    missing_paths = missing_template_paths(task, template_dir)
    groups = missing_template_groups(task, template_dir)

    assert missing_paths == [
        template_dir / "missing_success.png",
        template_dir / "missing_reset.png",
        tmp_path / "missing_spawn.png",
    ]

    assert groups == [
        MissingTemplateGroup(
            node="navigate:route:success",
            candidates=[template_dir / "missing_success.png"],
            required_count=1,
            existing_count=0,
        ),
        MissingTemplateGroup(
            node="navigate:route:reset",
            candidates=[template_dir / "missing_reset.png"],
            required_count=1,
            existing_count=0,
        ),
        MissingTemplateGroup(
            node="navigate:route:keyframes",
            candidates=[tmp_path / "missing_spawn.png"],
            required_count=1,
            existing_count=0,
        ),
    ]


def test_format_missing_template_groups_is_grouped_by_node(tmp_path: Path):
    groups = [
        MissingTemplateGroup(
            node="start_game",
            candidates=[tmp_path / "start_game.png"],
            required_count=1,
            existing_count=0,
        ),
        MissingTemplateGroup(
            node="close_popups",
            candidates=[tmp_path / "close_a.png", tmp_path / "close_b.png"],
            required_count=1,
            existing_count=0,
        ),
    ]

    text = format_missing_template_groups(groups)

    assert "缺少模板阶段 2 个" in text
    assert "node=start_game" in text
    assert "start_game.png" in text
    assert "需要至少 1 个" in text

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from navigation.route_config import load_navigation_route
from task_engine import TaskDefinition


@dataclass(frozen=True)
class MissingTemplateGroup:
    node: str
    candidates: list[Path]
    required_count: int = 1
    existing_count: int = 0


def resolve_template_path(template: str, template_dir: str | Path) -> Path:
    path = Path(template)
    if path.is_absolute() or path.exists():
        return path
    return Path(template_dir) / path


def missing_template_paths(task: TaskDefinition, template_dir: str | Path) -> list[Path]:
    missing: list[Path] = []
    seen: set[Path] = set()
    for node in task.nodes:
        template_names = node.template_names()
        if node.approach:
            template_names.extend(node.approach.target_templates)
            template_names.extend(node.approach.success_templates)
            if node.approach.interrupt_templates:
                template_names.extend(node.approach.interrupt_templates)
            if node.approach.joystick_templates:
                template_names.extend(node.approach.joystick_templates)
        if node.maturity and node.maturity.anchor_template:
            template_names.append(node.maturity.anchor_template)
        if node.maturity and node.maturity.tomorrow_template:
            template_names.append(node.maturity.tomorrow_template)
        if node.maturity and node.maturity.farm_templates:
            template_names.extend(node.maturity.farm_templates)
        if node.route:
            route_path = Path(node.route.path)
            if route_path not in seen:
                seen.add(route_path)
                if not route_path.exists():
                    missing.append(route_path)
                    continue
            try:
                route = load_navigation_route(route_path)
            except Exception:
                missing.append(route_path)
                continue
            route_paths: list[Path] = []
            route_paths.extend(resolve_template_path(template_name, template_dir) for template_name in route.success.templates)
            route_paths.extend(resolve_template_path(template_name, template_dir) for template_name in route.interrupt.templates)
            route_paths.extend(resolve_template_path(template_name, template_dir) for template_name in route.reset.templates)
            route_paths.extend(route.resolve_path(keyframe.image) for keyframe in route.keyframes)
            for path in route_paths:
                if path in seen:
                    continue
                seen.add(path)
                if not path.exists():
                    missing.append(path)
        for template_name in template_names:
            path = resolve_template_path(template_name, template_dir)
            if path in seen:
                continue
            seen.add(path)
            if not path.exists():
                missing.append(path)
    return missing


def missing_template_groups(task: TaskDefinition, template_dir: str | Path) -> list[MissingTemplateGroup]:
    groups: list[MissingTemplateGroup] = []
    for node in task.nodes:
        template_names = node.template_names()
        if template_names:
            candidates = [resolve_template_path(template_name, template_dir) for template_name in template_names]
            existing_count = sum(1 for path in candidates if path.exists())
            required_count = min(node.min_matches, len(candidates))
            if existing_count < required_count:
                groups.append(
                    MissingTemplateGroup(
                        node=node.name,
                        candidates=candidates,
                        required_count=required_count,
                        existing_count=existing_count,
                    )
                )
        if node.approach:
            groups.extend(
                _missing_candidate_group(
                    node=f"{node.name}:target",
                    templates=node.approach.target_templates,
                    template_dir=template_dir,
                )
            )
            groups.extend(
                _missing_candidate_group(
                    node=f"{node.name}:success",
                    templates=node.approach.success_templates,
                    template_dir=template_dir,
                )
            )
            if node.approach.interrupt_templates:
                groups.extend(
                    _missing_candidate_group(
                        node=f"{node.name}:interrupt",
                        templates=node.approach.interrupt_templates,
                        template_dir=template_dir,
                    )
                )
            if node.approach.joystick_templates:
                groups.extend(
                    _missing_candidate_group(
                        node=f"{node.name}:joystick",
                        templates=node.approach.joystick_templates,
                        template_dir=template_dir,
                    )
                )
        if node.maturity and node.maturity.anchor_template:
            groups.extend(
                _missing_candidate_group(
                    node=f"{node.name}:maturity_anchor",
                    templates=[node.maturity.anchor_template],
                    template_dir=template_dir,
                )
            )
        if node.maturity and node.maturity.tomorrow_template:
            groups.extend(
                _missing_candidate_group(
                    node=f"{node.name}:maturity_tomorrow",
                    templates=[node.maturity.tomorrow_template],
                    template_dir=template_dir,
                )
            )
        if node.maturity and node.maturity.farm_templates:
            groups.extend(
                _missing_candidate_group(
                    node=f"{node.name}:maturity_farm",
                    templates=node.maturity.farm_templates,
                    template_dir=template_dir,
                    required_count=node.maturity.farm_min_matches,
                )
            )
        if node.route:
            groups.extend(_missing_route_groups(node.name, node.route.path, template_dir))
    return groups


def _missing_route_groups(node_name: str, route_path_value: str, template_dir: str | Path) -> list[MissingTemplateGroup]:
    route_path = Path(route_path_value)
    if not route_path.exists():
        return [
            MissingTemplateGroup(
                node=f"{node_name}:route",
                candidates=[route_path],
                required_count=1,
                existing_count=0,
            )
        ]
    try:
        route = load_navigation_route(route_path)
    except Exception:
        return [
            MissingTemplateGroup(
                node=f"{node_name}:route",
                candidates=[route_path],
                required_count=1,
                existing_count=0,
            )
        ]

    groups: list[MissingTemplateGroup] = []
    if route.success.templates:
        groups.extend(
            _missing_candidate_group(
                node=f"{node_name}:route:success",
                templates=route.success.templates,
                template_dir=template_dir,
            )
        )
    if route.interrupt.templates:
        groups.extend(
            _missing_candidate_group(
                node=f"{node_name}:route:interrupt",
                templates=route.interrupt.templates,
                template_dir=template_dir,
            )
        )
    if route.reset.templates:
        groups.extend(
            _missing_candidate_group(
                node=f"{node_name}:route:reset",
                templates=route.reset.templates,
                template_dir=template_dir,
            )
        )
    keyframe_paths = [route.resolve_path(keyframe.image) for keyframe in route.keyframes]
    missing_keyframes = [path for path in keyframe_paths if not path.exists()]
    if missing_keyframes:
        groups.append(
            MissingTemplateGroup(
                node=f"{node_name}:route:keyframes",
                candidates=keyframe_paths,
                required_count=len(keyframe_paths),
                existing_count=len(keyframe_paths) - len(missing_keyframes),
            )
        )
    return groups


def _missing_candidate_group(
    *,
    node: str,
    templates: list[str],
    template_dir: str | Path,
    required_count: int = 1,
) -> list[MissingTemplateGroup]:
    candidates = [resolve_template_path(template_name, template_dir) for template_name in templates]
    existing_count = sum(1 for path in candidates if path.exists())
    required_count = min(required_count, len(candidates))
    if existing_count >= required_count:
        return []
    return [
        MissingTemplateGroup(
            node=node,
            candidates=candidates,
            required_count=required_count,
            existing_count=existing_count,
        )
    ]


def format_missing_template_groups(groups: list[MissingTemplateGroup], limit: int = 8) -> str:
    if not groups:
        return ""
    lines = [f"缺少模板阶段 {len(groups)} 个："]
    for group in groups[:limit]:
        lines.append(f"- node={group.node}（已有 {group.existing_count} 个，需要至少 {group.required_count} 个）")
        for candidate in group.candidates[:3]:
            lines.append(f"  * {candidate}")
        if len(group.candidates) > 3:
            lines.append(f"  * ... 还有 {len(group.candidates) - 3} 个候选")
    if len(groups) > limit:
        lines.append(f"- ... 还有 {len(groups) - limit} 个阶段")
    return "\n".join(lines)


def format_missing_templates(missing: list[Path], limit: int = 8) -> str:
    if not missing:
        return ""
    lines = [f"缺少模板文件 {len(missing)} 个："]
    for path in missing[:limit]:
        lines.append(f"- {path}")
    if len(missing) > limit:
        lines.append(f"- ... 还有 {len(missing) - limit} 个")
    return "\n".join(lines)

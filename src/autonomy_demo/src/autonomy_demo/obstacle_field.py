"""Utility helpers for parsing obstacles and performing ray intersections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from tf_conversions import transformations


@dataclass
class RaycastResult:
    """Container describing CPU raycast outputs."""

    distances: np.ndarray
    normals: np.ndarray
    hit_mask: np.ndarray
    hit_types: np.ndarray
    hit_indices: np.ndarray


@dataclass
class TorchRaycastResult:
    """Container describing torch-based raycast outputs."""

    distances: "torch.Tensor"
    normals: "torch.Tensor"
    hit_mask: "torch.Tensor"
    hit_types: "torch.Tensor"
    hit_indices: "torch.Tensor"


class ObstacleField:
    """Stores obstacle geometry and accelerates ray intersection queries."""

    def __init__(self) -> None:
        self.sphere_centers: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.sphere_radii: np.ndarray = np.empty((0,), dtype=np.float32)
        self.sphere_bounds: np.ndarray = np.empty((0,), dtype=np.float32)
        self.box_centers: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.box_half_extents: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.box_rotations: np.ndarray = np.empty((0, 3, 3), dtype=np.float32)
        self.box_inv_rotations: np.ndarray = np.empty((0, 3, 3), dtype=np.float32)
        self.box_bounding_radii: np.ndarray = np.empty((0,), dtype=np.float32)
        self.max_candidates: int = 512

        self._torch = None
        self._device = None
        self.sphere_centers_t = None
        self.sphere_radii_t = None
        self.box_centers_t = None
        self.box_half_extents_t = None
        self.box_rotations_t = None
        self.box_inv_rotations_t = None

    @property
    def supports_torch(self) -> bool:
        return (
            self._torch is not None
            and self.sphere_centers_t is not None
            and self.sphere_radii_t is not None
            and self.box_centers_t is not None
            and self.box_half_extents_t is not None
            and self.box_rotations_t is not None
            and self.box_inv_rotations_t is not None
        )

    def distance_to_point(self, point: Sequence[float]) -> float:
        """Return the minimum signed distance from *point* to any obstacle surface."""

        point_arr = np.asarray(point, dtype=np.float32)
        if point_arr.shape != (3,):
            point_arr = point_arr.reshape(3)

        distances: List[np.ndarray] = []

        if self.sphere_centers.size:
            diff = self.sphere_centers - point_arr
            if self.sphere_radii.size:
                radii = self.sphere_radii
            else:
                radii = np.zeros(diff.shape[0], dtype=np.float32)
            sphere_dist = np.linalg.norm(diff, axis=1) - radii
            distances.append(sphere_dist)

        if self.box_centers.size:
            rel = point_arr[None, :] - self.box_centers
            local = np.einsum("nij,nj->ni", self.box_inv_rotations, rel)
            excess = np.abs(local) - self.box_half_extents
            outside = np.maximum(excess, 0.0)
            outside_dist = np.linalg.norm(outside, axis=1)
            inside = np.minimum(np.max(excess, axis=1), 0.0)
            box_dist = outside_dist + inside
            distances.append(box_dist)

        if not distances:
            return float("inf")

        stacked = np.concatenate(distances)
        if stacked.size == 0:
            return float("inf")
        return float(np.min(stacked))

    def update_from_markers(
        self,
        markers: Sequence,
        use_torch: bool = False,
        torch_module=None,
        device=None,
    ) -> None:
        """Parse a sequence of visualization markers into cached geometry."""

        sphere_centers: List[Tuple[float, float, float]] = []
        sphere_radii: List[float] = []
        box_centers: List[Tuple[float, float, float]] = []
        box_sizes: List[Tuple[float, float, float]] = []
        box_rotations: List[np.ndarray] = []
        box_inv_rotations: List[np.ndarray] = []

        for marker in markers:
            center = (
                float(marker.pose.position.x),
                float(marker.pose.position.y),
                float(marker.pose.position.z),
            )
            size = (
                max(float(marker.scale.x), 1e-3),
                max(float(marker.scale.y), 1e-3),
                max(float(marker.scale.z), 1e-3),
            )
            quat = (
                float(marker.pose.orientation.x),
                float(marker.pose.orientation.y),
                float(marker.pose.orientation.z),
                float(marker.pose.orientation.w),
            )
            marker_type = getattr(marker, "type", 0)
            sphere_type = getattr(marker, "SPHERE", 2)
            if marker_type == sphere_type:
                sphere_centers.append(center)
                sphere_radii.append(size[0] / 2.0)
            else:
                rotation = self._quaternion_to_matrix(quat)
                box_centers.append(center)
                box_sizes.append(size)
                box_rotations.append(rotation)
                box_inv_rotations.append(rotation.T)

        self.sphere_centers = (
            np.asarray(sphere_centers, dtype=np.float32)
            if sphere_centers
            else np.empty((0, 3), dtype=np.float32)
        )
        self.sphere_radii = (
            np.asarray(sphere_radii, dtype=np.float32)
            if sphere_radii
            else np.empty((0,), dtype=np.float32)
        )
        self.sphere_bounds = (
            self.sphere_radii.copy()
            if self.sphere_radii.size
            else np.empty((0,), dtype=np.float32)
        )
        self.box_centers = (
            np.asarray(box_centers, dtype=np.float32)
            if box_centers
            else np.empty((0, 3), dtype=np.float32)
        )
        if box_sizes:
            sizes = np.asarray(box_sizes, dtype=np.float32)
            self.box_half_extents = sizes / 2.0
        else:
            self.box_half_extents = np.empty((0, 3), dtype=np.float32)
        self.box_rotations = (
            np.asarray(box_rotations, dtype=np.float32)
            if box_rotations
            else np.empty((0, 3, 3), dtype=np.float32)
        )
        self.box_inv_rotations = (
            np.asarray(box_inv_rotations, dtype=np.float32)
            if box_inv_rotations
            else np.empty((0, 3, 3), dtype=np.float32)
        )
        self.box_bounding_radii = (
            np.linalg.norm(self.box_half_extents, axis=1)
            if self.box_half_extents.size
            else np.empty((0,), dtype=np.float32)
        )

        if use_torch and torch_module is not None:
            self._torch = torch_module
            self._device = device
            torch_dtype = torch_module.float32
            if self.sphere_centers.size:
                self.sphere_centers_t = torch_module.from_numpy(self.sphere_centers).to(
                    device=device, dtype=torch_dtype
                )
                self.sphere_radii_t = torch_module.from_numpy(self.sphere_radii).to(
                    device=device, dtype=torch_dtype
                )
            else:
                self.sphere_centers_t = torch_module.empty((0, 3), device=device, dtype=torch_dtype)
                self.sphere_radii_t = torch_module.empty((0,), device=device, dtype=torch_dtype)
            if self.box_centers.size:
                self.box_centers_t = torch_module.from_numpy(self.box_centers).to(
                    device=device, dtype=torch_dtype
                )
                self.box_half_extents_t = torch_module.from_numpy(self.box_half_extents).to(
                    device=device, dtype=torch_dtype
                )
                self.box_rotations_t = torch_module.from_numpy(self.box_rotations).to(
                    device=device, dtype=torch_dtype
                )
                self.box_inv_rotations_t = torch_module.from_numpy(self.box_inv_rotations).to(
                    device=device, dtype=torch_dtype
                )
            else:
                self.box_centers_t = torch_module.empty((0, 3), device=device, dtype=torch_dtype)
                self.box_half_extents_t = torch_module.empty((0, 3), device=device, dtype=torch_dtype)
                self.box_rotations_t = torch_module.empty((0, 3, 3), device=device, dtype=torch_dtype)
                self.box_inv_rotations_t = torch_module.empty((0, 3, 3), device=device, dtype=torch_dtype)
        else:
            self._torch = None
            self._device = None
            self.sphere_centers_t = None
            self.sphere_radii_t = None
            self.box_centers_t = None
            self.box_half_extents_t = None
            self.box_rotations_t = None
            self.box_inv_rotations_t = None

    def _cull_indices(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        origin: np.ndarray,
        max_range: float,
    ) -> np.ndarray:
        if centers.size == 0:
            return np.empty((0,), dtype=np.int32)

        origin_vec = origin.astype(np.float32)
        diff = centers - origin_vec[np.newaxis, :]
        dist_sq = np.sum(diff * diff, axis=1)

        radii_array = np.asarray(radii, dtype=np.float32)
        center_count = centers.shape[0]
        if radii_array.size == 0:
            radii_array = np.zeros((center_count,), dtype=np.float32)
        elif radii_array.shape[0] != center_count:
            if radii_array.size > center_count:
                radii_array = radii_array[:center_count]
            else:
                pad = np.zeros((center_count - radii_array.size,), dtype=np.float32)
                if radii_array.size:
                    pad.fill(float(radii_array[-1]))
                radii_array = np.concatenate([radii_array, pad])

        reach = max_range + radii_array
        reach_sq = reach * reach
        mask = dist_sq <= reach_sq
        if not np.any(mask):
            return np.empty((0,), dtype=np.int32)

        indices = np.nonzero(mask)[0]
        if self.max_candidates > 0 and indices.size > self.max_candidates:
            order = np.argsort(dist_sq[indices])
            indices = indices[order[: self.max_candidates]]
        return indices.astype(np.int32)

    def cast_rays_cpu(
        self, origin: np.ndarray, directions: np.ndarray, max_range: float
    ) -> RaycastResult:
        """Intersect a bundle of rays with all stored geometry using NumPy."""

        num_rays = directions.shape[0]
        best_dist = np.full(num_rays, np.inf, dtype=np.float32)
        hit_mask = np.zeros(num_rays, dtype=bool)
        hit_types = np.full(num_rays, -1, dtype=np.int8)
        hit_indices = np.full(num_rays, -1, dtype=np.int32)
        normals = np.zeros((num_rays, 3), dtype=np.float32)

        sphere_dist = np.full(num_rays, np.inf, dtype=np.float32)
        sphere_idx = np.zeros(num_rays, dtype=np.int32)
        sphere_hit_mask = np.zeros(num_rays, dtype=bool)
        sphere_subset = None
        if self.sphere_centers.size:
            subset = self._cull_indices(
                self.sphere_centers, self.sphere_bounds, origin, max_range
            )
            if subset.size:
                centers = self.sphere_centers[subset]
                radii = self.sphere_radii[subset]
                sphere_dist, sphere_idx, sphere_hit_mask = self._intersect_spheres_cpu(
                    origin, directions, centers, radii, max_range
                )
                sphere_subset = subset

        update_mask = sphere_hit_mask & (sphere_dist < best_dist)
        if np.any(update_mask):
            best_dist[update_mask] = sphere_dist[update_mask]
            hit_mask[update_mask] = True
            hit_types[update_mask] = 0
            if sphere_subset is not None:
                global_idx = sphere_subset[sphere_idx]
                hit_indices[update_mask] = global_idx[update_mask]
            else:
                hit_indices[update_mask] = sphere_idx[update_mask]

        box_dist = np.full(num_rays, np.inf, dtype=np.float32)
        box_idx = np.zeros(num_rays, dtype=np.int32)
        box_hit_mask = np.zeros(num_rays, dtype=bool)
        box_subset = None
        box_subset_arrays = None
        if self.box_centers.size:
            subset = self._cull_indices(
                self.box_centers, self.box_bounding_radii, origin, max_range
            )
            if subset.size:
                centers = self.box_centers[subset]
                half_extents = self.box_half_extents[subset]
                rotations = self.box_rotations[subset]
                inv_rotations = self.box_inv_rotations[subset]
                box_dist, box_idx, box_hit_mask = self._intersect_boxes_cpu(
                    origin,
                    directions,
                    centers,
                    half_extents,
                    rotations,
                    inv_rotations,
                    max_range,
                )
                box_subset = subset
                box_subset_arrays = (centers, half_extents, rotations, inv_rotations)

        update_mask = box_hit_mask & (box_dist < best_dist)
        if np.any(update_mask):
            best_dist[update_mask] = box_dist[update_mask]
            hit_mask[update_mask] = True
            hit_types[update_mask] = 1
            if box_subset is not None:
                global_idx = box_subset[box_idx]
                hit_indices[update_mask] = global_idx[update_mask]
            else:
                hit_indices[update_mask] = box_idx[update_mask]

        if np.any(hit_mask):
            sphere_hit = hit_mask & (hit_types == 0)
            if np.any(sphere_hit) and sphere_subset is not None:
                idx = hit_indices[sphere_hit]
                dist = best_dist[sphere_hit]
                centers = self.sphere_centers[idx]
                hit_points = origin + directions[sphere_hit] * dist[:, None]
                normals[sphere_hit] = self._normalize_rows(hit_points - centers)

            box_hit = hit_mask & (hit_types == 1)
            if np.any(box_hit) and box_subset_arrays is not None:
                centers, half_extents, rotations, inv_rotations = box_subset_arrays
                local_idx = box_idx[box_hit]
                dist = best_dist[box_hit]
                centers_sel = centers[local_idx]
                inv_rot_sel = inv_rotations[local_idx]
                rot_sel = rotations[local_idx]
                half_sel = half_extents[local_idx]
                diff = origin[np.newaxis, :].astype(np.float32) - centers_sel
                local_origin = np.einsum("bi,bij->bj", diff, inv_rot_sel)
                dirs_sel = directions[box_hit]
                local_dir = np.einsum("ri,rij->rj", dirs_sel, inv_rot_sel)
                local_hit = local_origin + local_dir * dist[:, None]
                abs_diff = np.abs(np.abs(local_hit) - half_sel)
                axis = np.argmin(abs_diff, axis=1)
                normal_local = np.zeros_like(local_hit)
                normal_local[np.arange(normal_local.shape[0]), axis] = np.sign(
                    local_hit[np.arange(local_hit.shape[0]), axis]
                )
                normal_world = np.einsum("bij,bj->bi", rot_sel, normal_local)
                normals[box_hit] = self._normalize_rows(normal_world.astype(np.float32))

        return RaycastResult(best_dist, normals, hit_mask, hit_types, hit_indices)

    def cast_rays_torch(
        self,
        torch_module,
        device,
        origin: "torch.Tensor",
        directions: "torch.Tensor",
        max_range: float,
    ) -> TorchRaycastResult:
        """Intersect rays using torch on the requested device."""

        torch = torch_module
        num_rays = directions.shape[0]
        dtype = directions.dtype
        best_dist = torch.full((num_rays,), float("inf"), device=device, dtype=dtype)
        hit_mask = torch.zeros(num_rays, dtype=torch.bool, device=device)
        hit_types = torch.full((num_rays,), -1, dtype=torch.int8, device=device)
        hit_indices = torch.full((num_rays,), -1, dtype=torch.int64, device=device)
        normals = torch.zeros((num_rays, 3), device=device, dtype=dtype)

        origin_np = origin.detach().cpu().numpy()

        sphere_dist = torch.full((num_rays,), float("inf"), device=device, dtype=dtype)
        sphere_idx = torch.zeros(num_rays, dtype=torch.int64, device=device)
        sphere_hit_mask = torch.zeros(num_rays, dtype=torch.bool, device=device)
        sphere_subset = None
        if self.sphere_centers_t is not None and self.sphere_centers_t.numel() > 0:
            subset_np = self._cull_indices(
                self.sphere_centers, self.sphere_bounds, origin_np, max_range
            )
            if subset_np.size:
                subset = torch.from_numpy(subset_np).to(device=device, dtype=torch.long)
                centers = self.sphere_centers_t.index_select(0, subset)
                radii = self.sphere_radii_t.index_select(0, subset)
                sphere_dist, sphere_idx, sphere_hit_mask = self._intersect_spheres_torch(
                    torch, device, origin, directions, centers, radii, max_range
                )
                sphere_subset = subset

        update_mask = sphere_hit_mask & (sphere_dist < best_dist)
        if torch.any(update_mask):
            best_dist[update_mask] = sphere_dist[update_mask]
            hit_mask[update_mask] = True
            hit_types[update_mask] = 0
            if sphere_subset is not None:
                global_idx = sphere_subset[sphere_idx]
                hit_indices[update_mask] = global_idx[update_mask]
            else:
                hit_indices[update_mask] = sphere_idx[update_mask]

        box_dist = torch.full((num_rays,), float("inf"), device=device, dtype=dtype)
        box_idx = torch.zeros(num_rays, dtype=torch.int64, device=device)
        box_hit_mask = torch.zeros(num_rays, dtype=torch.bool, device=device)
        box_subset = None
        box_subset_arrays = None
        if self.box_centers_t is not None and self.box_centers_t.numel() > 0:
            subset_np = self._cull_indices(
                self.box_centers, self.box_bounding_radii, origin_np, max_range
            )
            if subset_np.size:
                subset = torch.from_numpy(subset_np).to(device=device, dtype=torch.long)
                centers = self.box_centers_t.index_select(0, subset)
                half_extents = self.box_half_extents_t.index_select(0, subset)
                rotations = self.box_rotations_t.index_select(0, subset)
                inv_rotations = self.box_inv_rotations_t.index_select(0, subset)
                box_dist, box_idx, box_hit_mask = self._intersect_boxes_torch(
                    torch,
                    device,
                    origin,
                    directions,
                    centers,
                    half_extents,
                    rotations,
                    inv_rotations,
                    max_range,
                )
                box_subset = subset
                box_subset_arrays = (centers, half_extents, rotations, inv_rotations)

        update_mask = box_hit_mask & (box_dist < best_dist)
        if torch.any(update_mask):
            best_dist[update_mask] = box_dist[update_mask]
            hit_mask[update_mask] = True
            hit_types[update_mask] = 1
            if box_subset is not None:
                global_idx = box_subset[box_idx]
                hit_indices[update_mask] = global_idx[update_mask]
            else:
                hit_indices[update_mask] = box_idx[update_mask]

        if torch.any(hit_mask):
            sphere_hit = hit_mask & (hit_types == 0)
            if torch.any(sphere_hit) and self.sphere_centers_t is not None:
                idx = hit_indices[sphere_hit]
                dist = best_dist[sphere_hit]
                centers = self.sphere_centers_t.index_select(0, idx)
                hit_points = origin + directions[sphere_hit] * dist.unsqueeze(1)
                normals[sphere_hit] = self._normalize_rows_torch(
                    torch, hit_points - centers
                )

            box_hit = hit_mask & (hit_types == 1)
            if torch.any(box_hit) and box_subset_arrays is not None:
                centers, half_extents, rotations, inv_rotations = box_subset_arrays
                local_idx = box_idx[box_hit]
                dist = best_dist[box_hit]
                centers_sel = centers.index_select(0, local_idx)
                inv_rot_sel = inv_rotations.index_select(0, local_idx)
                rot_sel = rotations.index_select(0, local_idx)
                half_sel = half_extents.index_select(0, local_idx)
                diff = origin.unsqueeze(0) - centers_sel
                local_origin = torch.einsum("bi,bij->bj", diff, inv_rot_sel)
                dirs_sel = directions[box_hit]
                local_dir = torch.einsum("ri,rij->rj", dirs_sel, inv_rot_sel)
                local_hit = local_origin + local_dir * dist.unsqueeze(1)
                abs_diff = torch.abs(torch.abs(local_hit) - half_sel)
                axis = torch.argmin(abs_diff, dim=1)
                normal_local = torch.zeros_like(local_hit)
                row_indices = torch.arange(normal_local.shape[0], device=device)
                normal_local[row_indices, axis] = torch.sign(local_hit[row_indices, axis])
                normal_world = torch.einsum("bij,bj->bi", rot_sel, normal_local)
                normals[box_hit] = self._normalize_rows_torch(torch, normal_world)

        return TorchRaycastResult(best_dist, normals, hit_mask, hit_types, hit_indices)

    # ------------------------------------------------------------------
    # CPU helpers
    # ------------------------------------------------------------------
    def _intersect_spheres_cpu(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        centers: np.ndarray,
        radii: np.ndarray,
        max_range: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if centers.size == 0:
            infs = np.full(directions.shape[0], np.inf, dtype=np.float32)
            zeros = np.zeros_like(infs, dtype=np.int32)
            hits = np.zeros_like(infs, dtype=bool)
            return infs, zeros, hits

        oc = origin - centers
        b = 2.0 * directions.dot(oc.T)
        c = np.sum(oc * oc, axis=1) - radii * radii
        discriminant = b * b - 4.0 * c
        mask = discriminant >= 0.0
        if not np.any(mask):
            infs = np.full(directions.shape[0], np.inf, dtype=np.float32)
            idx = np.zeros_like(infs, dtype=np.int32)
            return infs, idx, np.zeros_like(infs, dtype=bool)
        sqrt_disc = np.sqrt(np.clip(discriminant, 0.0, None))
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        t_candidate = np.where(t1 > 0.0, t1, t2)
        hit_matrix = mask & (t_candidate > 0.0)
        t_candidate = np.where(hit_matrix, t_candidate, np.inf)
        t_candidate = np.where(t_candidate <= max_range, t_candidate, np.inf)
        distances = np.min(t_candidate, axis=1)
        indices = np.argmin(t_candidate, axis=1)
        hit_mask = np.isfinite(distances)
        return distances.astype(np.float32), indices.astype(np.int32), hit_mask

    def _intersect_boxes_cpu(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        centers: np.ndarray,
        half_extents: np.ndarray,
        rotations: np.ndarray,
        inv_rotations: np.ndarray,
        max_range: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if centers.size == 0:
            infs = np.full(directions.shape[0], np.inf, dtype=np.float32)
            zeros = np.zeros_like(infs, dtype=np.int32)
            hits = np.zeros_like(infs, dtype=bool)
            return infs, zeros, hits

        diff = origin - centers
        local_origin = np.einsum("bi,bij->bj", diff, inv_rotations)
        local_dir = np.einsum("ri,bij->rbj", directions, inv_rotations)
        half = half_extents[np.newaxis, :, :]
        local_origin_b = local_origin[np.newaxis, :, :]

        abs_dir = np.abs(local_dir)
        zero_mask = abs_dir <= 1e-6
        inv_dir = np.empty_like(local_dir)
        inv_dir[~zero_mask] = 1.0 / local_dir[~zero_mask]
        inv_dir[zero_mask] = 0.0

        t1 = (-half - local_origin_b) * inv_dir
        t2 = (half - local_origin_b) * inv_dir

        t1[zero_mask] = -np.inf
        t2[zero_mask] = np.inf

        outside_mask = zero_mask & (np.abs(local_origin_b) > half)
        t1[outside_mask] = np.inf
        t2[outside_mask] = -np.inf

        t_lower = np.minimum(t1, t2)
        t_upper = np.maximum(t1, t2)
        t_near = np.max(t_lower, axis=2)
        t_far = np.min(t_upper, axis=2)
        t_candidate = np.where(t_near > 0.0, t_near, t_far)
        valid = (t_far >= t_near) & (t_candidate > 0.0) & (t_candidate <= max_range)
        t_candidate = np.where(valid, t_candidate, np.inf)

        distances = np.min(t_candidate, axis=1)
        indices = np.argmin(t_candidate, axis=1)
        hit_mask = np.isfinite(distances)
        return (
            distances.astype(np.float32),
            indices.astype(np.int32),
            hit_mask,
        )

    # ------------------------------------------------------------------
    # Torch helpers
    # ------------------------------------------------------------------
    def _intersect_spheres_torch(
        self,
        torch,
        device,
        origin: "torch.Tensor",
        directions: "torch.Tensor",
        centers: "torch.Tensor",
        radii: "torch.Tensor",
        max_range: float,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        if centers.numel() == 0:
            distances = torch.full(
                (directions.shape[0],), float("inf"), device=device, dtype=directions.dtype
            )
            indices = torch.zeros_like(distances, dtype=torch.int64)
            hits = torch.zeros_like(distances, dtype=torch.bool)
            return distances, indices, hits

        oc = origin - centers
        b = 2.0 * directions @ oc.t()
        c = torch.sum(oc * oc, dim=1) - radii * radii
        discriminant = b * b - 4.0 * c
        mask = discriminant >= 0.0
        if not torch.any(mask):
            distances = torch.full(
                (directions.shape[0],), float("inf"), device=device, dtype=directions.dtype
            )
            indices = torch.zeros_like(distances, dtype=torch.int64)
            return distances, indices, torch.zeros_like(distances, dtype=torch.bool)
        sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        t_candidate = torch.where(t1 > 0.0, t1, t2)
        inf_tensor = torch.full_like(t_candidate, float("inf"))
        t_candidate = torch.where(mask & (t_candidate > 0.0), t_candidate, inf_tensor)
        t_candidate = torch.where(t_candidate <= max_range, t_candidate, inf_tensor)
        distances, indices = torch.min(t_candidate, dim=1)
        hit_mask = torch.isfinite(distances)
        return distances, indices, hit_mask

    def _intersect_boxes_torch(
        self,
        torch,
        device,
        origin: "torch.Tensor",
        directions: "torch.Tensor",
        centers: "torch.Tensor",
        half_extents: "torch.Tensor",
        rotations: "torch.Tensor",
        inv_rotations: "torch.Tensor",
        max_range: float,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        if centers.numel() == 0:
            distances = torch.full(
                (directions.shape[0],), float("inf"), device=device, dtype=directions.dtype
            )
            indices = torch.zeros_like(distances, dtype=torch.int64)
            hits = torch.zeros_like(distances, dtype=torch.bool)
            return distances, indices, hits

        diff = origin - centers
        local_origin = torch.einsum("bi,bij->bj", diff, inv_rotations)
        local_dir = torch.einsum("ri,bij->rbj", directions, inv_rotations)
        half = half_extents.unsqueeze(0)
        local_origin_b = local_origin.unsqueeze(0)

        abs_dir = torch.abs(local_dir)
        zero_mask = abs_dir <= 1e-6
        inv_dir = torch.empty_like(local_dir)
        inv_dir[~zero_mask] = 1.0 / local_dir[~zero_mask]
        inv_dir[zero_mask] = 0.0

        t1 = (-half - local_origin_b) * inv_dir
        t2 = (half - local_origin_b) * inv_dir

        neg_inf = torch.full_like(local_dir, float("-inf"))
        pos_inf = torch.full_like(local_dir, float("inf"))
        t1 = torch.where(zero_mask, neg_inf, t1)
        t2 = torch.where(zero_mask, pos_inf, t2)

        outside_mask = zero_mask & (torch.abs(local_origin_b) > half)
        t1 = torch.where(outside_mask, pos_inf, t1)
        t2 = torch.where(outside_mask, neg_inf, t2)

        t_lower = torch.minimum(t1, t2)
        t_upper = torch.maximum(t1, t2)
        t_near = torch.max(t_lower, dim=2).values
        t_far = torch.min(t_upper, dim=2).values
        t_candidate = torch.where(t_near > 0.0, t_near, t_far)
        valid = (t_far >= t_near) & (t_candidate > 0.0) & (t_candidate <= max_range)
        inf_tensor = torch.full_like(t_candidate, float("inf"))
        t_candidate = torch.where(valid, t_candidate, inf_tensor)
        distances, indices = torch.min(t_candidate, dim=1)
        hit_mask = torch.isfinite(distances)
        return distances, indices, hit_mask





    def snapshot(self) -> dict:
        """Return a serializable dictionary describing the current obstacle state."""

        return {
            "sphere_centers": self.sphere_centers.copy(),
            "sphere_radii": self.sphere_radii.copy(),
            "box_centers": self.box_centers.copy(),
            "box_half_extents": self.box_half_extents.copy(),
            "box_rotations": self.box_rotations.copy(),
        }

    def load_snapshot(self, snapshot: dict) -> None:
        """Populate the field from a snapshot dictionary."""

        def _maybe(name: str, fallback: np.ndarray) -> np.ndarray:
            value = snapshot.get(name)
            if value is None:
                return fallback.copy()
            arr = np.asarray(value, dtype=np.float32)
            return arr.copy()

        self.sphere_centers = _maybe("sphere_centers", np.empty((0, 3), dtype=np.float32))
        self.sphere_radii = _maybe("sphere_radii", np.empty((0,), dtype=np.float32))
        self.sphere_bounds = (
            self.sphere_radii.copy() if self.sphere_radii.size else np.empty((0,), dtype=np.float32)
        )
        self.box_centers = _maybe("box_centers", np.empty((0, 3), dtype=np.float32))
        self.box_half_extents = _maybe("box_half_extents", np.empty((0, 3), dtype=np.float32))
        self.box_rotations = _maybe("box_rotations", np.empty((0, 3, 3), dtype=np.float32))
        if self.box_rotations.size:
            self.box_inv_rotations = np.transpose(self.box_rotations, (0, 2, 1))
        else:
            self.box_inv_rotations = np.empty((0, 3, 3), dtype=np.float32)
        self.box_bounding_radii = (
            np.linalg.norm(self.box_half_extents, axis=1)
            if self.box_half_extents.size
            else np.empty((0,), dtype=np.float32)
        )
        self._torch = None
        self._device = None
        self.sphere_centers_t = None
        self.sphere_radii_t = None
        self.box_centers_t = None
        self.box_half_extents_t = None
        self.box_rotations_t = None
        self.box_inv_rotations_t = None

    @staticmethod
    def _normalize_rows(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, eps, None)
        return vectors / norms

    @staticmethod
    def _normalize_rows_torch(torch, vectors: "torch.Tensor", eps: float = 1e-8):
        norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=eps)
        return vectors / norms

    @staticmethod
    def _quaternion_to_matrix(quat: Iterable[float]) -> np.ndarray:
        if abs(sum(q * q for q in quat)) <= 1e-12:
            return np.identity(3, dtype=np.float32)
        matrix = transformations.quaternion_matrix(quat)
        return matrix[0:3, 0:3].astype(np.float32)



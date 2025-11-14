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
        self.box_centers: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.box_half_extents: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.box_rotations: np.ndarray = np.empty((0, 3, 3), dtype=np.float32)
        self.box_inv_rotations: np.ndarray = np.empty((0, 3, 3), dtype=np.float32)

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

        if self.sphere_centers.size:
            sphere_result = self._intersect_spheres_cpu(origin, directions, max_range)
            sphere_dist, sphere_idx, sphere_hit_mask = sphere_result
            update_mask = sphere_hit_mask & (sphere_dist < best_dist)
            if np.any(update_mask):
                best_dist[update_mask] = sphere_dist[update_mask]
                hit_mask[update_mask] = True
                hit_types[update_mask] = 0
                hit_indices[update_mask] = sphere_idx[update_mask]

        box_aux = None
        if self.box_centers.size:
            box_result = self._intersect_boxes_cpu(origin, directions, max_range)
            box_dist, box_idx, box_hit_mask, box_aux = box_result
            update_mask = box_hit_mask & (box_dist < best_dist)
            if np.any(update_mask):
                best_dist[update_mask] = box_dist[update_mask]
                hit_mask[update_mask] = True
                hit_types[update_mask] = 1
                hit_indices[update_mask] = box_idx[update_mask]

        if np.any(hit_mask):
            sphere_hit = hit_mask & (hit_types == 0)
            if np.any(sphere_hit):
                idx = hit_indices[sphere_hit]
                dist = best_dist[sphere_hit]
                centers = self.sphere_centers[idx]
                hit_points = origin + directions[sphere_hit] * dist[:, None]
                normals[sphere_hit] = self._normalize_rows(hit_points - centers)

            box_hit = hit_mask & (hit_types == 1)
            if np.any(box_hit) and box_aux is not None:
                local_origin, local_dir = box_aux
                idx = hit_indices[box_hit]
                dist = best_dist[box_hit]
                local_origin_sel = local_origin[idx]
                # local_dir has shape (num_rays, num_boxes, 3)
                ld = local_dir[box_hit, idx, :]
                local_hit = local_origin_sel + ld * dist[:, None]
                half_extents = self.box_half_extents[idx]
                abs_diff = np.abs(np.abs(local_hit) - half_extents)
                axis = np.argmin(abs_diff, axis=1)
                normal_local = np.zeros_like(local_hit)
                normal_local[np.arange(normal_local.shape[0]), axis] = np.sign(
                    local_hit[np.arange(local_hit.shape[0]), axis]
                )
                rotations = self.box_rotations[idx]
                normal_world = np.einsum("mij,mj->mi", rotations, normal_local)
                normals[box_hit] = self._normalize_rows(normal_world)

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
        inf = torch.tensor(float("inf"), device=device, dtype=dtype)
        best_dist = torch.full((num_rays,), float("inf"), device=device, dtype=dtype)
        hit_mask = torch.zeros(num_rays, dtype=torch.bool, device=device)
        hit_types = torch.full((num_rays,), -1, dtype=torch.int8, device=device)
        hit_indices = torch.full((num_rays,), -1, dtype=torch.int64, device=device)
        normals = torch.zeros((num_rays, 3), device=device, dtype=dtype)

        sphere_data = None
        if self.sphere_centers_t is not None and self.sphere_centers_t.numel() > 0:
            sphere_data = self._intersect_spheres_torch(
                torch, device, origin, directions, max_range
            )
            sphere_dist, sphere_idx, sphere_hit_mask = sphere_data
            update_mask = sphere_hit_mask & (sphere_dist < best_dist)
            if torch.any(update_mask):
                best_dist[update_mask] = sphere_dist[update_mask]
                hit_mask[update_mask] = True
                hit_types[update_mask] = 0
                hit_indices[update_mask] = sphere_idx[update_mask]

        box_data = None
        if self.box_centers_t is not None and self.box_centers_t.numel() > 0:
            box_data = self._intersect_boxes_torch(
                torch, device, origin, directions, max_range
            )
            box_dist, box_idx, box_hit_mask, box_aux = box_data
            update_mask = box_hit_mask & (box_dist < best_dist)
            if torch.any(update_mask):
                best_dist[update_mask] = box_dist[update_mask]
                hit_mask[update_mask] = True
                hit_types[update_mask] = 1
                hit_indices[update_mask] = box_idx[update_mask]

        if torch.any(hit_mask):
            if sphere_data is not None:
                sphere_dist, _, _ = sphere_data
            if box_data is not None:
                _, _, _, (local_origin, local_dir) = box_data

            sphere_hit = hit_mask & (hit_types == 0)
            if torch.any(sphere_hit) and sphere_data is not None:
                idx = hit_indices[sphere_hit]
                dist = best_dist[sphere_hit]
                centers = self.sphere_centers_t[idx]
                hit_points = origin + directions[sphere_hit] * dist.unsqueeze(1)
                normals[sphere_hit] = self._normalize_rows_torch(
                    torch, hit_points - centers
                )

            box_hit = hit_mask & (hit_types == 1)
            if torch.any(box_hit) and box_data is not None:
                idx = hit_indices[box_hit]
                dist = best_dist[box_hit]
                local_origin_sel = local_origin[idx]
                ld = local_dir[box_hit]
                ld = ld[torch.arange(ld.shape[0], device=device), idx, :]
                local_hit = local_origin_sel + ld * dist.unsqueeze(1)
                half_extents = self.box_half_extents_t[idx]
                abs_diff = torch.abs(torch.abs(local_hit) - half_extents)
                axis = torch.argmin(abs_diff, dim=1)
                normal_local = torch.zeros_like(local_hit)
                row_indices = torch.arange(normal_local.shape[0], device=device)
                normal_local[row_indices, axis] = torch.sign(local_hit[row_indices, axis])
                rotations = self.box_rotations_t[idx]
                normal_world = torch.einsum("mij,mj->mi", rotations, normal_local)
                normals[box_hit] = self._normalize_rows_torch(torch, normal_world)

        return TorchRaycastResult(best_dist, normals, hit_mask, hit_types, hit_indices)

    # ------------------------------------------------------------------
    # CPU helpers
    # ------------------------------------------------------------------
    def _intersect_spheres_cpu(
        self, origin: np.ndarray, directions: np.ndarray, max_range: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        oc = origin - self.sphere_centers
        b = 2.0 * directions.dot(oc.T)
        c = np.sum(oc * oc, axis=1) - self.sphere_radii * self.sphere_radii
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
        self, origin: np.ndarray, directions: np.ndarray, max_range: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        diff = origin - self.box_centers
        local_origin = np.einsum("bi,bij->bj", diff, self.box_inv_rotations)
        local_dir = np.einsum("ri,bij->rbj", directions, self.box_inv_rotations)
        half = self.box_half_extents[np.newaxis, :, :]
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
            (local_origin, local_dir),
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
        max_range: float,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        oc = origin - self.sphere_centers_t
        b = 2.0 * directions @ oc.t()
        c = torch.sum(oc * oc, dim=1) - self.sphere_radii_t * self.sphere_radii_t
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
        max_range: float,
    ) -> Tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        Tuple["torch.Tensor", "torch.Tensor"],
    ]:
        diff = origin - self.box_centers_t
        local_origin = torch.einsum("bi,bij->bj", diff, self.box_inv_rotations_t)
        local_dir = torch.einsum("ri,bij->rbj", directions, self.box_inv_rotations_t)
        half = self.box_half_extents_t.unsqueeze(0)
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
        return distances, indices, hit_mask, (local_origin, local_dir)

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


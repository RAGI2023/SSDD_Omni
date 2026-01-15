from torch.utils.data import Dataset
import os
import torch
import cv2
import numpy as np

try:
    from equirect_utils import equirect_to_fisheye_ucm, DEFAULT_JITTER_CONFIG, NO_JITTER_CONFIG
except ImportError:
    from utils.equirect_utils import equirect_to_fisheye_ucm, DEFAULT_JITTER_CONFIG, NO_JITTER_CONFIG


class EquiDataset(Dataset):
    """
    Returns:
        imgs: Tensor [Nviews, 3, Hc, Wc], float32 in [0,1]
        img_original: Tensor [3, Hc, Wc], float32 in [0,1]
    """

    def __init__(self, folder_path, canvas_size=(1920, 960), jitter_cfg=None, **kwargs):
        self.folder_path = folder_path
        self.jitter_cfg = jitter_cfg or {}
        self.canvas_size = canvas_size  # (Wc, Hc)

        self.image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {folder_path}.")

        # Output fisheye size (each view)
        self.out_w = int(kwargs.get("out_w", 512))
        self.out_h = int(kwargs.get("out_h", 512))

        # UCM parameters (recommended)
        # Pass f_pix explicitly if possible; else fallback to deprecated fov_diag_deg.
        self.f_pix = kwargs.get("f_pix", None)          # e.g. 220.0
        self.xi = float(kwargs.get("xi", 0.9))          # e.g. 0.9
        self.mask_mode = kwargs.get("mask_mode", "inscribed")  # "inscribed"|"diagonal"|"none"

        # Deprecated fallback (if f_pix is None)
        self.fov = float(kwargs.get("fov", 130))

        # Replace old KB k_jitter with UCM jitter:
        # We keep config key name "k_jitter" for backward compatibility.
        # Interpretation:
        #   strength ~ Uniform(1-k_jitter, 1+k_jitter)
        #   xi'  = xi  * strength_xi   (clamped)
        #   f'   = f   / strength_f    (optional, keeps FOV variation)
        #
        # You can control behavior via kwargs:
        #   xi_jitter: float or None   (overrides k_jitter for xi)
        #   f_jitter:  float or None   (overrides k_jitter for f)
        #   jitter_target: "xi"|"f"|"both"
        #
        cfg_kjit = self.jitter_cfg.get("k_jitter", 0.0)
        self.xi_jitter = kwargs.get("xi_jitter", cfg_kjit)
        self.f_jitter = kwargs.get("f_jitter", 0.0)  # default: do NOT jitter f unless you set it
        self.jitter_target = kwargs.get("jitter_target", "xi")  # default: only jitter xi

        # Reasonable clamps to avoid invalid UCM regimes
        self.xi_min = float(kwargs.get("xi_min", 0.05))
        self.xi_max = float(kwargs.get("xi_max", 3.0))
        self.f_min = float(kwargs.get("f_min", 10.0))

        # Canonical directions (guaranteed correct)
        # Order matters because you lay them into the canvas by index.
        self.VIEWS = [
            ("front", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            ("right", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("back",  np.array([0.0, 0.0, -1.0], dtype=np.float32)),
            ("left",  np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
            # ("top",    np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            # ("bottom", np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        ]

    def __len__(self):
        return len(self.image_files)

    def _sample_strength(self, jitter):
        jitter = float(jitter) if jitter is not None else 0.0
        if jitter <= 0:
            return 1.0
        return float(np.random.uniform(1.0 - jitter, 1.0 + jitter))

    def _sample_ucm_params(self):
        """
        Produce per-sample UCM params (xi, f_pix) with jitter.
        - xi jitter: multiplicative
        - f_pix jitter: (optional) inverse multiplicative so that increasing strength means wider FOV
        """
        xi = float(self.xi)

        if self.f_pix is None:
            f_pix = None
        else:
            f_pix = float(self.f_pix)

        # Jitter xi and/or f_pix
        if self.jitter_target in ("xi", "both"):
            s_xi = self._sample_strength(self.xi_jitter)
            xi = float(np.clip(xi * s_xi, self.xi_min, self.xi_max))

        if self.jitter_target in ("f", "both") and (f_pix is not None):
            s_f = self._sample_strength(self.f_jitter)
            # invert scaling: higher strength -> smaller f -> wider view
            f_pix = max(self.f_min, float(f_pix / s_f))

        return xi, f_pix

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Sample per-item UCM params
        xi, f_pix = self._sample_ucm_params()

        # Build kwargs for current equirect_to_fisheye_ucm
        ucm_kwargs = dict(
            xi=xi,
            mask_mode=self.mask_mode,
        )
        if f_pix is not None:
            ucm_kwargs["f_pix"] = float(f_pix)
        else:
            # fallback to deprecated fov approx (if your equirect utils supports it)
            ucm_kwargs["fov_diag_deg"] = float(self.fov)

        # Generate views in the EXACT order of self.VIEWS
        # Expected order: front(0), right(1), back(2), left(3)
        views = []
        for _, base_dir in self.VIEWS:
            out = equirect_to_fisheye_ucm(
                img,
                out_w=self.out_w,
                out_h=self.out_h,
                base_dir=base_dir,
                # keep 0 here; jitter_cfg can inject micro-rotations internally
                yaw_deg=0.0,
                pitch_deg=0.0,
                roll_deg=0.0,
                jitter_cfg=self.jitter_cfg,
                **ucm_kwargs,
            )
            # Ensure size is consistent (safety)
            if out.shape[0] != self.out_h or out.shape[1] != self.out_w:
                out = cv2.resize(out, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            views.append(out)

        # Stack -> [4, H, W, 3] then to tensor [4, 3, H, W] in [0,1]
        views_np = np.stack(views, axis=0).astype(np.uint8)  # [4, H, W, 3]
        imgs = torch.from_numpy(views_np).permute(0, 3, 1, 2).float() / 255.0  # [4, 3, H, W]

        # Original panorama resized to canvas_size (kept as before)
        Wc, Hc = int(self.canvas_size[0]), int(self.canvas_size[1])
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        img_original = torch.nn.functional.interpolate(
            img_t.unsqueeze(0), size=(Hc, Wc), mode="bilinear", align_corners=False
        ).squeeze(0).to(imgs.device)  # [3, Hc, Wc]

        return imgs, img_original


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = EquiDataset(
        folder_path="/data/360SP-data/eval",
        canvas_size=(1024, 512),
        out_w=512,
        out_h=512,
        jitter_cfg=DEFAULT_JITTER_CONFIG,

        # Recommended UCM params:
        f_pix=220.0,
        xi=0.9,
        mask_mode="inscribed",

        # Replace old k_jitter behavior:
        # Use existing jitter_cfg["k_jitter"] as xi jitter by default.
        # Optional fine control:
        xi_jitter=0.2,          # or omit to use jitter_cfg["k_jitter"]
        f_jitter=0.0,           # set >0 if you want FOV variation via f_pix
        jitter_target="xi",     # "xi"|"f"|"both"
    )

    print("Dataset length:", len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    write_img = True
    for i, (imgs, img_original) in enumerate(loader):
        print("Batch", i, "images shape:", imgs.shape)
        if write_img:
            dirname = "runs/test_output/"
            os.makedirs(dirname, exist_ok=True)
            imgs_np = imgs.numpy()
            for b in range(imgs_np.shape[0]):
                for v, (view_name, _) in enumerate(dataset.VIEWS):
                    cv2.imwrite(
                        f"{dirname}test_{view_name}_{i}_{b}.png",
                        cv2.cvtColor(
                            (imgs_np[b, v] * 255).astype(np.uint8).transpose(1, 2, 0),
                            cv2.COLOR_RGB2BGR,
                        ),
                    )
            cv2.imwrite(
                f"{dirname}test_original_{i}.png",
                cv2.cvtColor(
                    (img_original[0].numpy() * 255).astype(np.uint8).transpose(1, 2, 0),
                    cv2.COLOR_RGB2BGR,
                ),
            )
        break

    print("Data loading test completed.")

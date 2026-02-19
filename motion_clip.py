import numpy as np
import torch
import torch.nn as nn

class MotionClip(nn.Module):
    """
    Wrappe un encodeur de mouvement et le projette dans l'espace d'embedding texte (Qwen).
    Supporte désormais la Cross-Attention via les text_tokens.
    """

    def __init__(self, motion_model, text_dim: int, text_model_name: str | None = None):
        super().__init__()
        self.motion_model = motion_model
        self.text_dim = int(text_dim)
        self.text_model_name = text_model_name  # bookkeeping optionnel

        self.proj = None
        self._in_dim = None

        # Température CLIP (échelle logarithmique)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, motions, text_tokens=None, **motion_kwargs):
        """
        Args:
            motions: Tenseur de mouvement paddé (B, T, D).
            text_tokens: Tenseur des tokens de texte (B, N, E) pour la Cross-Attention.
            motion_kwargs: Arguments additionnels (ex: lengths=...).
        
        Returns:
            Tenseur projeté (B, text_dim) prêt pour la CLIP loss.
        """
        # On passe les motions ET les tokens à l'encodeur interne (SOTA/InterGen style)
        # C'est ici que la magie de la Cross-Attention opère
        z = self.motion_model(motions, text_tokens=text_tokens, **motion_kwargs) # (B, M)
        
        if z.ndim != 2:
            raise ValueError(f"motion_model must return (B, M). Got {z.shape}.")

        # Initialisation "Lazy" de la couche de projection finale (vers l'espace Qwen)
        if self._in_dim is None:
            self._in_dim = int(z.shape[1])
            if self._in_dim == self.text_dim:
                self.proj = nn.Identity()
            else:
                # Si ton encodeur sort du 512 ou 1024 et Qwen attend 3584
                self.proj = nn.Linear(self._in_dim, self.text_dim)
            self.proj = self.proj.to(z.device)

        return self.proj(z)
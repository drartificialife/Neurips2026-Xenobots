"""
Multi-head P2I Network Architecture
Prompt-to-Intervention network with shared backbone and behavior-specific heads
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class P2INetwork(nn.Module):
    """Multi-head P2I network: shared CNN backbone + 8 independent output heads"""

    def __init__(self, num_behaviors=8, embedding_dim=384, hidden_dim=16):
        super().__init__()
        self.num_behaviors = num_behaviors

        # Shared backbone: Conv1d cascade
        self.backbone = nn.Sequential(
            nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Multi-head: 8 independent heads, one per behavior
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(num_behaviors)
        ])

    def forward(self, embeddings, behavior_idx=None):
        """
        Forward pass

        Args:
            embeddings: (batch_size, embedding_dim) or (batch_size, 1, embedding_dim)
            behavior_idx: which behavior head to use (0-7), or None for all

        Returns:
            output: (batch_size, 1) if behavior_idx specified, else (batch_size, num_behaviors)
        """
        # Reshape if needed: (batch, dim) -> (batch, 1, dim)
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)

        # Pass through backbone
        features = self.backbone(embeddings)  # (batch, hidden_dim, 1)
        features = self.pool(features)  # (batch, hidden_dim, 1)
        features = features.squeeze(-1)  # (batch, hidden_dim)

        # Pass through heads
        if behavior_idx is not None:
            # Single head
            output = self.heads[behavior_idx](features)
            return output
        else:
            # All heads
            outputs = []
            for head in self.heads:
                outputs.append(head(features))
            return torch.cat(outputs, dim=1)  # (batch, num_behaviors)


class PromptEncoder:
    """SentenceBERT encoder for prompts"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, prompts):
        """
        Encode prompts to embeddings

        Args:
            prompts: list of strings or single string

        Returns:
            embeddings: (batch_size, 384) tensor
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        embeddings = self.model.encode(prompts, convert_to_tensor=True)
        return embeddings

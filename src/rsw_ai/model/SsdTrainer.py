import torch
from torch.utils.data import DataLoader


class SsdTrainer:

    def train(
        self,
        train_dataset,
        model,
        optimizer,
        epochs=10,
    ):

        # ==================================================
        # Device
        # ==================================================

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print("Training on:", device)

        model.to(device)

        # ==================================================
        # DataLoader
        # ==================================================

        train_loader = DataLoader(
            train_dataset,
            batch_size=10,
            shuffle=True,
            collate_fn=lambda batch: tuple(zip(*batch))
        )

        # ==================================================
        # Training
        # ==================================================

        model.train()

        for epoch in range(epochs):

            print(f"Epoch {epoch + 1}/{epochs}")

            epoch_loss = 0.0

            for batch_idx, (images, targets) in enumerate(train_loader):

                # ==========================================
                # Image -> GPU
                # ==========================================

                images = [
                    image.to(device)
                    for image in images
                ]

                # ==========================================
                # Target -> GPU
                # ==========================================

                targets = [
                    {
                        "boxes": target["boxes"].to(device),
                        "labels": target["labels"].to(device),
                    }
                    for target in targets
                ]

                # ==========================================
                # Forward
                # ==========================================

                loss_dict = model(
                    images,
                    targets
                )

                loss = sum(loss_dict.values())

                # ==========================================
                # Backpropagation
                # ==========================================

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=10
                )

                optimizer.step()

                # ==========================================
                # Accumulate epoch loss
                # ==========================================

                epoch_loss += loss.item()

                # ==========================================
                # Print batch loss
                # ==========================================

                if batch_idx % 10 == 0:

                    print(
                        f"Batch {batch_idx}, "
                        f"Loss: {loss.item():.4f}"
                    )

            # ==================================================
            # Average Epoch Loss
            # ==================================================

            average_epoch_loss = (
                epoch_loss / len(train_loader)
            )

            print(
                f"Epoch Loss: {average_epoch_loss:.4f}"
            )

        # ==================================================
        # Save Model
        # ==================================================

        self.save_model(
            model,
            "/content/drive/MyDrive/RSW_Y2S1_AI/ssd_model.pth"
        )

        print(
            "SSD model saved to:"
            "/content/drive/MyDrive/RSW_Y2S1_AI/ssd_model.pth"
        )

    # ======================================================
    # Save Model
    # ======================================================

    def save_model(self, model, path):

        torch.save(
            model.state_dict(),
            path
        )
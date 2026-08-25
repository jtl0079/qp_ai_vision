import torch
from torch.utils.data import DataLoader


class SsdTrainer:
    """def save_model(self,model, path):
      torch.save(model.state_dict(), path)
"""
    def train(
        self,
        train_dataset,
        model,
        optimizer,
        epochs=10,
    ):

        # ==========================
        # Device
        # ==========================

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print("Training on:", device)


        # model -> GPU
        model.to(device)


        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=lambda batch: tuple(zip(*batch))
        )


        model.train()


        for epoch in range(epochs):

            print(f"Epoch {epoch+1}/{epochs}")

            for batch_idx, (images, targets) in enumerate(train_loader):

                # ==========================
                # image -> GPU
                # ==========================

                images = [
                    image.to(device)
                    for image in images
                ]


                # ==========================
                # target -> GPU
                # ==========================

                targets = [
                    {
                        "boxes": target["boxes"].to(device),
                        "labels": target["labels"].to(device),
                    }
                    for target in targets
                ]


                # ==========================
                # Forward
                # ==========================

                loss_dict = model(
                    images,
                    targets
                )


                loss = sum(loss_dict.values())


                optimizer.zero_grad()

                loss.backward()

                optimizer.step()


                if batch_idx % 10 == 0:
                    print(
                        f"Batch {batch_idx}, Loss: {loss.item():.4f}"
                    )


            print(
                f"Epoch Loss: {loss.item():.4f}"
            )
        ##self.save_model(model, "ssd_model.pth")
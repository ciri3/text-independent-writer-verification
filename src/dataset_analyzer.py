import numpy as np

from dataset import IAMDataset


dataset = IAMDataset(data_dir="../data", granularity="words")

widths = []
heights = []
aspect_ratios = []

for i in range(len(dataset)):
    sample = dataset[i]

    width, height = sample["image"].size

    widths.append(width)
    heights.append(height)
    aspect_ratios.append(width / height)


def print_metric(name, values):
    print(f"\n{name}")
    print(f"Min:    {np.min(values):.2f}")
    print(f"Mean:   {np.mean(values):.2f}")
    print(f"Median: {np.median(values):.2f}")
    print(f"90%:    {np.percentile(values, 90):.2f}")
    print(f"95%:    {np.percentile(values, 95):.2f}")
    print(f"99%:    {np.percentile(values, 99):.2f}")
    print(f"Max:    {np.max(values):.2f}")


print(f"Number of samples: {len(dataset)}")

print_metric("Width", widths)
print_metric("Height", heights)
print_metric("Aspect ratio", aspect_ratios)


TARGET_HEIGHT = 64
MAX_WIDTHS = [256, 320, 384, 448, 512]

resized_widths = [
    width * TARGET_HEIGHT / height
    for width, height in zip(widths, heights)
]

print_metric(
    "Width after proportional resize to H=64",
    resized_widths
)

resized_widths = np.array(resized_widths)


print("\nComparison of candidate max_W values:")

for max_width in MAX_WIDTHS:

    # Quali immagini supererebbero max_width dopo
    # essere state portate ad altezza 64?
    exceeding = resized_widths > max_width
    fitting = resized_widths <= max_width

    n_exceeding = np.sum(exceeding)
    pct_exceeding = n_exceeding / len(resized_widths) * 100

    # Immagini che entrano normalmente nella canvas
    fitting_widths = resized_widths[fitting]

    occupancy = fitting_widths / max_width
    padding = max_width - fitting_widths

    print(f"\nmax_W = {max_width}")
    print(f"Exceeding: {n_exceeding} ({pct_exceeding:.2f}%)")

    print(f"Mean padding: {np.mean(padding):.2f} px")
    print(f"Median padding: {np.median(padding):.2f} px")

    print(f"Mean occupancy: {np.mean(occupancy) * 100:.2f}%")
    print(f"Median occupancy: {np.median(occupancy) * 100:.2f}%")

    for threshold in [0.25, 0.50, 0.75]:
        pct = np.mean(occupancy < threshold) * 100

        print(
            f"Occupancy < {threshold:.0%}: "
            f"{pct:.2f}%"
        )

    # Analisi delle immagini troppo larghe
    if n_exceeding > 0:

        exceeding_widths = resized_widths[exceeding]

        # Quanto dobbiamo ulteriormente ridimensionarle
        # affinché la larghezza diventi max_width?
        scale_factors = max_width / exceeding_widths

        # Se riduciamo proporzionalmente anche l'altezza,
        # questa sarà l'altezza risultante.
        resulting_heights = TARGET_HEIGHT * scale_factors

        print(
            f"Median scale factor: "
            f"{np.median(scale_factors):.3f}"
        )

        print(
            f"5th percentile scale factor: "
            f"{np.percentile(scale_factors, 5):.3f}"
        )

        print(
            f"1st percentile scale factor: "
            f"{np.percentile(scale_factors, 1):.3f}"
        )

        print(
            f"Median resulting height: "
            f"{np.median(resulting_heights):.2f} px"
        )

        print(
            f"5th percentile resulting height: "
            f"{np.percentile(resulting_heights, 5):.2f} px"
        )

        print(
            f"1st percentile resulting height: "
            f"{np.percentile(resulting_heights, 1):.2f} px"
        )
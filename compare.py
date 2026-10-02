from src.inference import WriterVerifier


verifier = WriterVerifier()

result = verifier.compare(
    "im_inf/foto1.jpg",
    "im_inf/foto2.jpg"
)

print(f"Distanza: {result['distance']:.4f}")
print(f"Soglia: {result['threshold']:.4f}")
print(f"Verdetto: {result['prediction']}")
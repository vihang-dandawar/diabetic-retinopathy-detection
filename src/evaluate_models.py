import os
import tensorflow as tf
import config as cfg

# ============================================================
# GPU SETUP
# ============================================================
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

# ============================================================
# CUSTOM LOSS (needed if model was trained with focal loss)
# ============================================================
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        loss = -y_true * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)
        return tf.reduce_sum(alpha * loss, axis=-1)
    focal_loss.__name__ = "focal_loss"
    return focal_loss

# ============================================================
# MAIN EVALUATION
# ============================================================
def evaluate_all_models():

    # --------------------------------------------------------
    # TEST DATASET PATH
    # --------------------------------------------------------
    test_dir = os.path.join(cfg.BASE_DIR, "data_processed", "test")

    if not os.path.exists(test_dir):
        print("❌ Test folder not found:", test_dir)
        return

    print("📂 Loading test data from:", test_dir)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(224, 224),     # change to (300,300) for EfficientNetB3
        batch_size=16,
        label_mode="categorical",
        shuffle=False
    ).prefetch(tf.data.AUTOTUNE)

    # --------------------------------------------------------
    # MODEL FOLDER
    # --------------------------------------------------------
    models_dir = os.path.join(cfg.BASE_DIR, "models")

    if not os.path.exists(models_dir):
        print("❌ Models folder not found:", models_dir)
        return

    model_files = [
        f for f in os.listdir(models_dir)
        if f.endswith(".keras") or f.endswith(".h5")
    ]

    if not model_files:
        print("❌ No .keras or .h5 models found.")
        return

    results = {}

    print("\n" + "="*60)
    print("🏆 STARTING MODEL EVALUATION")
    print("="*60)

    # --------------------------------------------------------
    # LOOP THROUGH ALL MODELS
    # --------------------------------------------------------
    for model_file in model_files:

        model_path = os.path.join(models_dir, model_file)

        print(f"\n📌 Evaluating: {model_file}")

        tf.keras.backend.clear_session()

        try:
            # Load model safely
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    "focal_loss": categorical_focal_loss(),
                    "categorical_focal_loss": categorical_focal_loss()
                },
                compile=False
            )

            # Compile fresh
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            # Evaluate
            loss, acc = model.evaluate(test_ds, verbose=1)

            results[model_file] = acc

            print(f"✅ Accuracy: {acc*100:.2f}%")
            print(f"✅ Loss    : {loss:.4f}")

        except Exception as e:
            print(f"❌ Failed to evaluate {model_file}")
            print("Error:", e)

    # --------------------------------------------------------
    # FINAL RANKINGS
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("🥇 FINAL RANKINGS")
    print("="*60)

    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for i, (name, acc) in enumerate(ranked, start=1):
        print(f"{i}. {name:<35} {acc*100:.2f}%")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    evaluate_all_models()
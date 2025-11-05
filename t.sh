#!/bin/bash

# Zdefiniuj listę elementów. Możesz dowolnie modyfikować tę listę.
MOJA_LISTA=("spatial_Resnet152_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true" "spatial_DenseNet121_decoder_dim_512_fine_tune_encoder_false_fine_tune_embeddings_false" "spatial_DenseNet121_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true" "spatial_DenseNet161_decoder_dim_512_fine_tune_encoder_false_fine_tune_embeddings_false" "spatial_DenseNet161_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true" "spatial_DenseNet201_decoder_dim_512_fine_tune_encoder_false_fine_tune_embeddings_false" "spatial_DenseNet201_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true" "spatial_InceptionV3_decoder_dim_512_fine_tune_encoder_false_fine_tune_embeddings_false" "spatial_InceptionV3_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true" "spatial_Regnet16_decoder_dim_512_fine_tune_encoder_false_fine_tune_embeddings_false" "spatial_Regnet16_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true" "spatial_Resnet101_decoder_dim_512_fine_tune_encoder_false_fine_tune_embeddings_false" "spatial_Resnet101_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true" "spatial_Resnet152_decoder_dim_512_fine_tune_encoder_false_fine_tune_embeddings_false")
for element in "${MOJA_LISTA[@]}"
do
  # Sprawdzenie, czy element nie jest pusty
  if [ -n "$element" ]; then
    echo "Przetwarzanie elementu: $element"

    # Tworzenie katalogu głównego i podkatalogu 'checkpoints' za jednym razem.
    # Flaga '-p' tworzy całą ścieżkę, nawet jeśli katalogi nadrzędne nie istnieją.
    mkdir -p "$element/checkpoints"

    # Nadawanie uprawnień do edycji, odczytu i wykonywania (777 -> rwxrwxrwx)
    # dla nowo utworzonego katalogu i wszystkiego wewnątrz niego.
    # Flaga '-R' oznacza rekurencyjnie (dla katalogu i jego zawartości).
    chmod -R 777 "$element"

    echo "Utworzono strukturę katalogów i nadano uprawnienia dla: $element"
    echo "---"
  fi
done

echo "Zakończono."

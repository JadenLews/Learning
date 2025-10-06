 # Machine Learning Models Used

Tried several models, each with different strengths:

(a) Convolutional Neural Network (CNN)
	•	Think of it as a standard image classifier.
	•	Good at learning from local pixel patterns.
	•	Worked reasonably well, especially when inputs were simple or low-resolution.

(b) Autoencoder + CatBoost (ACB)
	•	Autoencoder: compresses images into dense “codes” (latent features).
	•	CatBoost: a gradient-boosted decision tree model that uses those features for classification.
	•	This hybrid was better at early predictions because it extracted robust features.

(c) PredRNN-V2 (Recurrent Neural Network for Videos)
	•	A special kind of RNN designed for spatiotemporal (space + time) data like videos.
	•	Forecasts future frames of the erosion channel.
	•	Good at short-term predictions but tended to drift or “hallucinate” thick channels over long horizons.

(d) Autoencoder + MLP for Forecasting
	•	Instead of predicting the next image directly, it predicted the latent code.
	•	More stable over long horizons, especially with binary data ￼.



# Terms
1. Latent code:
   1. sometimes called a latent vector or embedding
   2. compressed representation of your data learned by a model—often an autoencoder.
   3. You start with a big image (say, a 50×50 pixel erosion snapshot → 2,500 numbers).
      1. An encoder network squashes it into a small vector (say, 64 numbers).
   4. It’s called latent because it’s hidden—you don’t specify what those numbers mean. The network learns them automatically in a way that makes it easier to reconstruct the original input.
   5. Why use latent codes?
      1. Compression: They reduce high-dimensional inputs (like images) to lower-dimensional, information rich codes.
	2.	Feature Extraction: Latent codes capture important patterns in the data (like channel shapes or flow direction) while ignoring noise.
	3.	Easier Modeling: Instead of directly modeling full images (thousands of pixels), you can predict how the latent code changes over time—much simpler.

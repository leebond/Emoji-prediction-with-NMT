Unzip all contents in a named folder.

Make sure you have the following python packages
tensorboard==1.13.1
tensorflow==1.13.1
tensorflow-estimator==1.13.0
Keras==2.2.4
Keras-Applications==1.0.7
Keras-Preprocessing==1.0.9


Task 1
$ python 1a_en_cnn_train.py
Requires wiki-news-300d-1M.magnitude in static folder. Please download separately from Plasticity (http://magnitude.plasticity.ai/fasttext/light/wiki-news-300d-1M.magnitude).
This will generate static and submission files: en_cnn_model.h5, en_cnn_submission, en_dictionary. The submission here is the prediction on the training data.
$ python 1b_en_cnn_test.py
This will generate the submission file base on the test data.

Task 2
Requires wiki.es.magnitude (converted from wiki.es.vec) in static folder. Please download separately from FastText(https://fasttext.cc/docs/en/pretrained-vectors.html) and do the convertion.
$ python 2a_es_cnn_train.py
This will generate static and submission files: es_cnn_model.h5, es_cnn_submission, es_dictionary. The submission here is the prediction on the training data.
$ python 2b_es_cnn_test.py
This will generate the submission file base on the test data.

Task 3
Naive Method
Requires wiki-news-300d-1M.magnitude in static folder
$ python 3a_en+es_cnn_train.py
Accreditation ./static/en-es.txt (from https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.txt)
./static/es_to_en_text.pkl (translated text from Spanish to English)
$ python 3b_en+es_cnn_train.py

NMT method
Requires wiki-news-300d-1M.magnitude in static folder
./static/es_translated_list.pkl (translated text from Spanish to English using NMT model from https://www.tensorflow.org/alpha/tutorials/text/nmt_with_attention)
$ python 3.1a en+es_cnn_nmt_train.py
$ python 3.1b en+es_cnn_nmt_test.py

from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import YourFileUploadForm  
import os
from django.conf import settings
import joblib
from glob import glob
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from keras.models import load_model
import pickle
from scipy import signal
from scipy.fftpack import fft, ifft
import statsmodels.api as sm
import antropy as ant
from scipy.stats import kurtosis
from scipy.stats import skew
import tensorflow as tf
import pywt


def index(request):
    if request.method == 'POST':
        form = YourFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            eeg_files_path = os.path.join(settings.MEDIA_ROOT, 'eeg_files')
            existing_files = glob(os.path.join(eeg_files_path, '*'))
            for file in existing_files:
                os.remove(file)
                
            fdt_file = form.cleaned_data['csvFile']

            with open(os.path.join(settings.MEDIA_ROOT, 'eeg_files', fdt_file.name), 'wb+') as destination:
                for chunk in fdt_file.chunks():
                    destination.write(chunk)
            file_path = glob(r'pictures\eeg_files\*.csv')
            for file in file_path:
                df = pd.read_csv(file)

            mymodel = joblib.load(r"C:\Users\Dhruv\Desktop\Project\model\models\knn_EO.pkl")

            eeg_data = df.iloc[:, 1:].values

            feature_vectors = []

            for channel in range(eeg_data.shape[1]):
                channel_data = eeg_data[:, channel]

                perm_entropy = ant.perm_entropy(channel_data, normalize=True)
                spectral_entropy = ant.spectral_entropy(
                    channel_data, sf=100, method="welch", normalize=True
                )
                svd_entropy = ant.svd_entropy(channel_data, normalize=True)
                hjorth_params = ant.hjorth_params(channel_data)
                petrosian_fd = ant.petrosian_fd(channel_data)
                katz_fd = ant.katz_fd(channel_data)
                higuchi_fd = ant.higuchi_fd(channel_data)
                dfa = ant.detrended_fluctuation(channel_data)
                channel_skewness = skew(channel_data)
                channel_kurtosis = kurtosis(channel_data)

                feature_vector = [
                    perm_entropy,
                    spectral_entropy,
                    svd_entropy,
                    hjorth_params[0],
                    hjorth_params[1],
                    petrosian_fd,
                    katz_fd,
                    higuchi_fd,
                    dfa,
                    channel_skewness,
                    channel_kurtosis,
                ]

                feature_vectors.append(feature_vector)


            x_test = np.array(feature_vectors, dtype=object)

            predictions = mymodel.predict(x_test)

            percentage_healthy = np.mean(predictions == 0) * 100
            percentage_mdd = np.mean(predictions == 1) * 100

            if percentage_healthy > percentage_mdd:
                diagnosis = f"Prediction: Healthy with {percentage_healthy:.2f}% confidence"
            else:
                diagnosis = f"Prediction: MDD with {percentage_mdd:.2f}% confidence"
            

            return render(request, 'index.html', {'form': form, 'response': diagnosis})  
        else:
            return HttpResponseRedirect("Failed")
    else:
        form = YourFileUploadForm()

    return render(request, 'index.html', {'form': form, 'response': None})


def success(request):
    return render(request, 'success.html') 
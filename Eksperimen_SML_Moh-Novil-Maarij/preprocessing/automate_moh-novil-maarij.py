# Data Libraries
import pandas as pd    
import numpy as np     
import math            

# Graphing Libraries
import matplotlib.pyplot as plt    
import seaborn as sns              

# Preprocessing Libraries
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler     
from category_encoders import TargetEncoder  
import os

def preprocess_air_quality(input_file_path, output_file_path='air_quality_cleaned.csv', 
                          target_column='AQI', columns_to_drop=['Date', 'City'], 
                          outlier_method='iqr', iqr_multiplier=1.5, verbose=True):
    """
    Fungsi untuk preprocessing data kualitas udara
    
    Parameters:
    -----------
    input_file_path : str
        Path ke file CSV input
    output_file_path : str, default='air_quality_cleaned.csv'
        Path untuk menyimpan file hasil preprocessing
    target_column : str, default='AQI'
        Nama kolom target yang tidak akan di-scale
    columns_to_drop : list, default=['Date', 'City']
        List kolom yang akan dihapus
    outlier_method : str, default='iqr'
        Metode penghapusan outlier ('iqr' untuk IQR method)
    iqr_multiplier : float, default=1.5
        Multiplier untuk IQR dalam penentuan outlier
    verbose : bool, default=True
        Apakah menampilkan informasi proses
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame yang sudah dipreprocessing
    str
        Path file output yang tersimpan
    """
    
    try:
        # 1. Load data
        if verbose:
            print("📁 Memuat data...")
        air_quality = pd.read_csv(input_file_path)
        
        if verbose:
            print(f"✅ Data berhasil dimuat dengan ukuran: {air_quality.shape}")
            print(f"📊 Kolom: {list(air_quality.columns)}")
        
        # 2. Hapus outlier
        if verbose:
            print("\n🔍 Menghapus outlier...")
        
        # Ambil semua kolom numerik
        numerical_cols = air_quality.select_dtypes(include='number').columns
        
        if outlier_method == 'iqr':
            original_size = air_quality.shape[0]
            
            # Hapus outlier secara langsung pada air_quality
            for col in numerical_cols:
                Q1 = air_quality[col].quantile(0.25)
                Q3 = air_quality[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                # Filter baris dalam rentang tanpa outlier
                before_filter = air_quality.shape[0]
                air_quality = air_quality[(air_quality[col] >= lower_bound) & (air_quality[col] <= upper_bound)]
                
                if verbose and before_filter != air_quality.shape[0]:
                    removed = before_filter - air_quality.shape[0]
                    print(f"   • Kolom '{col}': menghapus {removed} outlier")
            
            if verbose:
                total_removed = original_size - air_quality.shape[0]
                print(f"✅ Total {total_removed} baris outlier dihapus")
                print(f"📏 Ukuran setelah hapus outlier: {air_quality.shape}")
        
        # 3. Hapus kolom yang tidak diperlukan
        if verbose:
            print(f"\n🗑️ Menghapus kolom: {columns_to_drop}")
        
        for col in columns_to_drop:
            if col in air_quality.columns:
                air_quality.drop(col, inplace=True, axis=1)
                if verbose:
                    print(f"   • Kolom '{col}' berhasil dihapus")
            elif verbose:
                print(f"   • Kolom '{col}' tidak ditemukan, dilewati")
        
        # 4. Standardisasi fitur (kecuali target)
        if verbose:
            print(f"\n⚖️ Melakukan standardisasi fitur (kecuali kolom target '{target_column}')...")
        
        # Update kolom numerik setelah penghapusan kolom
        numerical_cols = air_quality.select_dtypes(include='number').columns
        
        # Kecualikan kolom target
        if target_column in numerical_cols:
            fitur_input = numerical_cols.drop(target_column)
        else:
            fitur_input = numerical_cols
            if verbose:
                print(f"⚠️ Kolom target '{target_column}' tidak ditemukan dalam kolom numerik")
        
        # Inisialisasi scaler dan fit_transform ke fitur input saja
        ss = StandardScaler()
        air_quality[fitur_input] = ss.fit_transform(air_quality[fitur_input])
        
        if verbose:
            print(f"✅ Standardisasi berhasil untuk {len(fitur_input)} kolom fitur")
            print(f"📊 Kolom yang di-standardisasi: {list(fitur_input)}")
        
        # 5. Simpan hasil
        if verbose:
            print(f"\n💾 Menyimpan hasil ke: {output_file_path}")
        
        air_quality.to_csv(output_file_path, index=False)
        
        if verbose:
            print("✅ Preprocessing selesai!")
            print(f"📄 File tersimpan: {output_file_path}")
            print(f"📏 Ukuran akhir: {air_quality.shape}")
            print(f"🎯 Kolom akhir: {list(air_quality.columns)}")
        
        return air_quality, output_file_path
        
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file_path}' tidak ditemukan!")
        return None, None
    except Exception as e:
        print(f"❌ Error dalam preprocessing: {str(e)}")
        return None, None

audio = "D:\ICTC\Cocktail Party\Sources\sample_male.wav";
[~,fs] = audioread("D:\ICTC\Cocktail Party\Sources\sample_male.wav");
features = mfcc_new(audio,fs,3,130);
csvwrite("C:\Users\kapoo\Downloads\pipeline\output_mfcc.csv",features);

% Define file paths
csv_file = "C:\Users\kapoo\Downloads\pipeline\output_mfcc.csv";   % CSV file path
npy_file = "C:\Users\kapoo\Downloads\pipeline\output_mfcc.npy";   % Output NPY file path

% Read CSV file into MATLAB table
data_table = readtable(csv_file);

% Convert table to numeric matrix (assuming all numeric data)
data_matrix = table2array(data_table);

% Save as .npy for Python
save(npy_file, 'data_matrix', '-v7.3');  

disp("CSV converted to NumPy array and saved as .npy");

system("python C:\Users\kapoo\Downloads\pipeline\predict.py");
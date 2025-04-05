from modelgenerator.generator import *

if __name__ == "__main__":
    print("Generating a dataset")
    square_count = 1000
    incomplete_square_prct = 0.5
    noisy_square_prct = 0

    circle_count = 0
    incomplete_circle_prct = 0
    noisy_circle_prct = 0

    noise_count = 0
    lines_count = 0

    CreateDatasetFolders(squares=square_count, incomplete_s=incomplete_square_prct, noisy_s=noisy_square_prct, 
                         circle=circle_count, incomplete_c=incomplete_circle_prct, noisy_c=noisy_circle_prct,
                         noise=noise_count, lines=lines_count)
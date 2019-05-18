import library.image_processing.movie as movies
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a movie by passing arguments to ffmpeg')
    parser.add_argument('-i', dest='image', type=int, default=800,
                        help='Number of points to sample along the circulation contour')

    # PARSE COMMAND LINE INPUTS
    args = parser.parse_args()

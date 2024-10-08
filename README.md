# Spotify Visualization

### Authors
* Cat Reinhart (catalinar@vt.edu)
* Daniel Hassler (hassledw@vt.edu)

## Hosted Link
https://spotify-dashboard-7lja3sqida-ue.a.run.app/

## Background
With music being popular all around the world due to its 
powerful ability to represent creativity, soul, and expression, 
we find visualizing such features important. We wish to understand the 
foundation of musical patterns prevalent in popular tracks and highlight our 
individual preferences.  Both of us on the team have diverse musical backgrounds 
that we can bring to this project subject, allowing us to explore the creative aspects 
of the visualization process for music in unique ways. Since music is subjective, 
we hope to provide additional prospectives for our audience through a visual 
medium with interactive plots.

## Dataset
For our dataset, we have decided to visualize a Kaggle.com hosted dataset called Spotify Tracks, 
consisting of 114,000 songs with 20 unique features (9 categorical and 11 numeric).

## Access our Dashboard
Our dashboard is hosted by GCP on link: https://spotify-dashboard-7lja3sqida-ue.a.run.app/ (this is our `app.py` file)

Steps taken to deploy dashboard: 
```
docker build -f Dockerfile -t gcr.io/infovis-427015/spotify-dashboard:test .
docker push gcr.io/infovis-427015/spotify-dashboard:test
gcloud run deploy spotify-dashboard --image gcr.io/infovis-427015/spotify-dashboard:test
```

## How to Run
1. To run our dashboard locally, navigate to the `dev-app.py` file and click "run" in the PyCharm IDE.
2. To run our Phase 1 static plot code, navigate to `Phase 1.py` file and click "run" in the PyCharm IDE.
3. To run outlier detection and PCA analysis, navigate to `analysis_and_testing.py` and click "run" in the PyCharm IDE.
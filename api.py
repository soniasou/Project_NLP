from flask import Flask, request, jsonify
from annoy import AnnoyIndex

app = Flask(__name__)
dim_1= 1000
dim_2= 100

# Load the Annoy database
annoy_db_1 = AnnoyIndex(dim_1, metric='angular')  
                                            
annoy_db_2 = AnnoyIndex(dim_2, metric='angular')
annoy_db_2.load('index_1.ann')
annoy_db_1.load('index.ann')  
@app.route('/') 

@app.route('/reco', methods=['POST'])
def reco():
    vector = request.json['vector']
    recommendation_type = request.json['type']
    
    # Perform the recommendation based on the type
    if recommendation_type == 'bagword':
        reco= annoy_db_1.get_nns_by_vector(vector, 5)
    elif recommendation_type == 'glove':
        reco = annoy_db_2.get_nns_by_vector(vector, 5)
    return jsonify(reco)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 

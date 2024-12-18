from flask import Flask, request, jsonify, render_template
from quality_score import get_property_data, get_images, calculate_property_score

app = Flask(__name__)

property_data_cache = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get-property-data", methods=["POST"])
def fetch_property_data():
    global property_data_cache

    data = request.json
    address = data.get("address")

    if not address:
        return jsonify({"error": "Address is required"}), 400

    if address in property_data_cache:
        return jsonify({"message": "Data already cached", "property_data": property_data_cache[address]})

    try:
        property_data = get_property_data(address)
        if property_data:
            property_data_cache[address] = property_data
            relevant_fields = [
                "address",
                "yearBuilt",
                "bedrooms",
                "bathrooms",
                "description",
                "livingArea",
                "lotSize",
                "livingAreaUnitsShort",
                "price",
                "lastSoldPrice",
                "taxAssessedValue",
                "zestimate",
                "dateSoldString",
            ]
            relevant_property_data = {key: property_data[key] for key in relevant_fields if key in property_data}
            return jsonify({"property_data": relevant_property_data})
        else:
            return jsonify({"error": "Failed to fetch property data"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-property-images", methods=["POST"])
def fetch_property_images():
    global property_data_cache

    data = request.json
    address = data.get("address")

    if not address:
        return jsonify({"error": "Address is required"}), 400

    if address not in property_data_cache:
        return jsonify({"error": "Property data not found. Fetch property data first."}), 404

    try:
        property_data = property_data_cache[address]
        images = get_images(property_data)
        print(images)
        if images:
            image_urls = list(images.values())
            return jsonify({"images": image_urls})
        else:
            return jsonify({"error": "No images found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-quality-score", methods=["POST"])
def fetch_quality_score():
    global property_data_cache

    data = request.json
    address = data.get("address")

    if not address:
        return jsonify({"error": "Address is required"}), 400

    if address not in property_data_cache:
        return jsonify({"error": "Property data not found. Fetch property data first."}), 404

    try:
        property_data = property_data_cache[address]
        images = get_images(property_data)
        quality_score = calculate_property_score(images)
        return jsonify({"quality_score": quality_score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

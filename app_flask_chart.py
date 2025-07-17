from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load and train model
df = pd.read_csv("Laptop_Motherboard_Health_Monitoring_Dataset.csv")
X = df[["RAMUsage", "Temperature", "Voltage", "DiskUsage", "FanSpeed"]]
y = df["CPUUsage"]
model = LinearRegression()
model.fit(X, y)

def generate_scatter_plot():
    fig, ax = plt.subplots()
    ax.scatter(df["Temperature"], df["CPUUsage"], alpha=0.5, c='skyblue')
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("CPU Usage (%)")
    ax.set_title("Scatter Plot: Temperature vs CPU Usage")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

@app.route("/")
def index():
    plot = generate_scatter_plot()
    return render_template("index_chart.html", plot=plot)

@app.route("/predict", methods=["POST"])
def predict():
    ram = float(request.form["ram"])
    temp = float(request.form["temp"])
    volt = float(request.form["volt"])
    disk = float(request.form["disk"])
    fan = int(request.form["fan"])

    input_data = pd.DataFrame([[ram, temp, volt, disk, fan]], columns=X.columns)
    prediction = model.predict(input_data)[0]
    plot = generate_scatter_plot()
    return render_template("index_chart.html", prediction=round(prediction, 2), plot=plot)

if __name__ == "__main__":
    app.run(debug=True)

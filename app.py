from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import os
from fastapi.responses import Response

FEATURE_MAPPING = {
    "pages_viewed": [
        "pages_viewed",
        "page_views",
        "num_pages",
        "views",
        "Pages_Viewed",
    ],
    "cart_additions": ["cart_additions", "add_to_cart", "cart_events"],
    "session_duration": [
        "session_duration",
        "time_spent",
        "session_time",
        "session_duration_sec",
        "Session_Duration_Minutes",
        "Is_Returning_Customer",
    ],
    "past_purchase_count": ["past_purchase_count", "previous_orders", "order_count"],
    "is_returning_user": [
        "is_returning_user",
        "returning",
        "repeat_user",
        "returning_user",
    ],
}


def map_company_columns(df, mapping):
    mapped_df = pd.DataFrame()

    for model_col, possible_names in mapping.items():
        found = False
        for col in possible_names:
            if col in df.columns:
                mapped_df[model_col] = df[col]
                found = True
                break

        if not found:
            raise ValueError(f"Missing required feature for model: {model_col}")

    return mapped_df


app = FastAPI(title="Purchase Intent Prediction")

templates = Jinja2Templates(directory="templates")

# Loading  model
with open("purchase_intent_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None,
            "purchase_probability": None,
            "intent_segment": None,
            "recommended_action": None,
            "preview": None,
            "csv_ready": False,
            "error": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()

    pages_viewed = int(form["pages_viewed"])
    cart_additions = int(form["cart_additions"])
    session_duration = int(form["session_duration"])
    past_purchase_count = int(form["past_purchase_count"])
    is_returning_user = int(form["is_returning_user"])

    has_cart = 1 if cart_additions > 0 else 0
    engagement_score = pages_viewed * session_duration

    input_df = pd.DataFrame(
        [
            {
                "pages_viewed": pages_viewed,
                "cart_additions": cart_additions,
                "has_cart": has_cart,
                "session_duration": session_duration,
                "engagement_score": engagement_score,
                "past_purchase_count": past_purchase_count,
                "is_returning_user": is_returning_user,
            }
        ]
    )

    probability = model.predict_proba(input_df)[:, 1][0]

    if probability >= 0.7:
        intent = "High Intent"
        action = "Target with Offer"
    elif probability >= 0.4:
        intent = "Medium Intent"
        action = "Send Reminder"
    else:
        intent = "Low Intent"
        action = "No Marketing Spend"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "probability": round(probability, 2),
            "intent": intent,
            "action": action,
        },
    )


@app.post("/predict_csv", response_class=HTMLResponse)
async def predict_csv(request: Request, file: UploadFile = File(...)):
    raw_df = pd.read_csv(file.file)
    if raw_df.empty:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": "Uploaded CSV is empty."}
        )

    try:
        df = map_company_columns(raw_df, FEATURE_MAPPING)
    except ValueError as e:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": str(e)}
        )

    numeric_cols = [
        "pages_viewed",
        "cart_additions",
        "session_duration",
        "past_purchase_count",
        "is_returning_user",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isnull().any().any():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Some required columns contain non-numeric or missing values.",
            },
        )
    try:
        df["has_cart"] = (df["cart_additions"] > 0).astype(int)
        df["engagement_score"] = df["pages_viewed"] * df["session_duration"]

        model_features = [
            "pages_viewed",
            "cart_additions",
            "has_cart",
            "session_duration",
            "engagement_score",
            "past_purchase_count",
            "is_returning_user",
        ]

        probabilities = model.predict_proba(df[model_features])[:, 1]
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Data processing error: {str(e)}"},
        )

    df["purchase_probability"] = probabilities

    def get_intent(p):
        if p >= 0.7:
            return "High Intent"
        elif p >= 0.4:
            return "Medium Intent"
        else:
            return "Low Intent"

    def get_action(p):
        if p >= 0.7:
            return "Target with Offer"
        elif p >= 0.4:
            return "Send Reminder"
        else:
            return "No Marketing Spend"

    df["intent_segment"] = df["purchase_probability"].apply(get_intent)
    df["recommended_action"] = df["purchase_probability"].apply(get_action)

    output_path = "predictions.csv"
    df.to_csv(output_path, index=False)

    return templates.TemplateResponse(
    "index.html",
    {
        "request": request,
        "prediction": prediction,
        "purchase_probability": probability,
        "intent_segment": intent,
        "recommended_action": action,
        "preview": preview_html,
        "csv_ready": True,
        "error": None,
    },
)


@app.get("/download")
def download_file():
    return FileResponse(
        "predictions.csv",
        media_type="text/csv",
        filename="purchase_intent_predictions.csv",
    )


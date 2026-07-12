from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from us_visa.constants import APP_HOST, APP_PORT
from us_visa.pipline.prediction_pipeline import USvisaClassifier, USvisaData
from us_visa.pipline.training_pipeline import TrainPipeline


app = FastAPI(title="Visa Approval Prediction")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request = request

    async def get_usvisa_data(self):
        form = await self.request.form()
        self.continent = form.get("continent")
        self.education_of_employee = form.get("education_of_employee")
        self.has_job_experience = form.get("has_job_experience")
        self.requires_job_training = form.get("requires_job_training")
        self.no_of_employees = form.get("no_of_employees")
        self.company_age = form.get("company_age")
        self.region_of_employment = form.get("region_of_employment")
        self.prevailing_wage = form.get("prevailing_wage")
        self.unit_of_wage = form.get("unit_of_wage")
        self.full_time_position = form.get("full_time_position")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("usvisa.html", {"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        TrainPipeline().run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_usvisa_data()

        usvisa_data = USvisaData(
            continent=form.continent,
            education_of_employee=form.education_of_employee,
            has_job_experience=form.has_job_experience,
            requires_job_training=form.requires_job_training,
            no_of_employees=form.no_of_employees,
            company_age=form.company_age,
            region_of_employment=form.region_of_employment,
            prevailing_wage=form.prevailing_wage,
            unit_of_wage=form.unit_of_wage,
            full_time_position=form.full_time_position,
        )

        usvisa_df = usvisa_data.get_usvisa_input_data_frame()
        model_predictor = USvisaClassifier()
        value = model_predictor.predict(dataframe=usvisa_df)[0]
        status = "Visa-approved" if value == 1 else "Visa Not-Approved"

        return templates.TemplateResponse(
            "usvisa.html",
            {"request": request, "context": status},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "usvisa.html",
            {"request": request, "context": f"Error: {e}"},
        )


if __name__ == "__main__":
    app_run(app="app:app", host=APP_HOST, port=APP_PORT, reload=True)
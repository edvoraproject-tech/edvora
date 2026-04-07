import os
import json
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, Optional
from google import genai
from urllib.parse import urlparse

# ── Config ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CATALOG_PATH = "courses_catalog_CS_IT_IS_L1_L4.json"

client = genai.Client(api_key=GEMINI_API_KEY)

# ── Load catalog ────────────────────────────────────────────────────
if not os.path.exists(CATALOG_PATH):
    raise FileNotFoundError(f"Catalog not found: {CATALOG_PATH}")

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    catalog_list = json.load(f)

CATALOG = {c["courseCode"]: c for c in catalog_list}

ALLOWED_DOMAINS = {
    "ocw.mit.edu",
    "cs229.stanford.edu",
    "cs231n.stanford.edu",
    "web.stanford.edu",
    "cs144.github.io",
    "15445.courses.cs.cmu.edu",
    "openstax.org",
    "docs.python.org",
    "pandas.pydata.org",
    "kaggle.com",
    "mode.com",
    "owasp.org",
    "www.postgresql.org",
    "postgresql.org",
    "linuxcommand.org",
    "sre.google",
    "scrumguides.org",
    "www.netacad.com",
    "netacad.com",
    "projects.iq.harvard.edu",
    "d2l.ai",
    "course.fast.ai",
    "spinningup.openai.com",
    "www.elementsofai.com",
    "elementsofai.com",
    "jakevdp.github.io",
    "clauswilke.com",
    "incompleteideas.net",
    "davidstarsilver.wordpress.com",
    "www.crypto101.io",
    "crypto101.io",
    "www2.eecs.berkeley.edu",
    "edx.org",
    "www.edx.org",
    "explore.skillbuilder.aws",
    "docs.aws.amazon.com",
    "wiki.postgresql.org",
    "coursera.org",
}


def url_is_allowed(u: str) -> bool:
    if not u or not isinstance(u, str):
        return False
    try:
        p = urlparse(u.strip())
        if p.scheme not in ("https", "http"):
            return False
        host = (p.netloc or "").lower()
        host = host[4:] if host.startswith("www.") else host
        allowed = {d.replace("www.", "") for d in ALLOWED_DOMAINS}
        return host in allowed
    except Exception:
        return False


# ── Flask app ───────────────────────────────────────────────────────
app = Flask(__name__)

IRRELEVANT_MSG_AR = "نحن نرد فقط على الأسئلة المتعلقة بالمقررات المسجّل بها."

LearningStyle = Literal["video", "book", "mix"]
LevelEnum = Literal["beginner", "intermediate", "advanced"]
CurrentLevelEnum = Literal["beginner", "intermediate", "advanced"]


# ── Request models ──────────────────────────────────────────────────
class InputCourse(BaseModel):
    courseCode: str = Field(..., min_length=2)
    courseName: str = Field(..., min_length=2)
    skillLevel: LevelEnum = "intermediate"


class RoadmapLiteRequest(BaseModel):
    learning_style: LearningStyle
    academic_goal: str = Field(..., min_length=2)
    weeklyHours: float = Field(default=12, gt=0, le=80)
    currentLevel: CurrentLevelEnum = "intermediate"
    courses: List[InputCourse] = Field(..., min_length=1, max_length=10)


class ChatRequest(BaseModel):
    learning_style: LearningStyle
    academic_goal: str = Field(..., min_length=2)
    weeklyHours: float = Field(default=12, gt=0, le=80)
    currentLevel: CurrentLevelEnum = "intermediate"
    courses: List[InputCourse] = Field(..., min_length=1, max_length=10)
    question: str = Field(..., min_length=1)


# ── Response schemas ────────────────────────────────────────────────
class Resource(BaseModel):
    type: Literal["book", "video", "mix"]
    title: str
    url: str


class CourseLitePlan(BaseModel):
    courseCode: str
    courseName: str
    skillLevel: LevelEnum
    resource: Optional[Resource] = None
    plan_ar: List[str] = Field(..., min_length=5, max_length=5)
    note: Optional[str] = None


class RoadmapLiteResponse(BaseModel):
    learning_style: LearningStyle
    weeklyHours: float
    currentLevel: CurrentLevelEnum
    academic_goal: str
    courses: List[CourseLitePlan]


class GeminiResourcePick(BaseModel):
    resource: Optional[Resource] = None
    note: Optional[str] = None


# ── Catalog helpers ─────────────────────────────────────────────────
def pick_resource_from_catalog(course_obj: dict, learning_style: str) -> Optional[dict]:
    resources = course_obj.get("resources", [])
    exact = [r for r in resources if r.get("type") == learning_style and r.get("url")]
    if exact:
        return exact[0]
    if learning_style == "mix":
        for t in ("mix", "video", "book"):
            cand = [r for r in resources if r.get("type") == t and r.get("url")]
            if cand:
                return cand[0]
    for t in ("book", "video", "mix"):
        cand = [r for r in resources if r.get("type") == t and r.get("url")]
        if cand:
            return cand[0]
    return None


def gemini_pick_resource(courseName: str, learning_style: str) -> GeminiResourcePick:
    prompt = f"""
You must select ONE working study resource URL for the course.

ABSOLUTE RULES:
- Output MUST be valid JSON matching the schema.
- You MUST NOT invent or guess URLs.
- You MUST choose a URL whose domain is in this allowlist ONLY:
{sorted(ALLOWED_DOMAINS)}
- IMPORTANT: Do NOT return resource=null unless it is truly impossible to choose from the allowlist.
- Before returning null, you MUST attempt at least 3 different plausible options from the allowlist domains.

SELECTION STRATEGY (do this internally):
1) Prefer well-known course hubs in the allowlist (ocw.mit.edu, web.stanford.edu, openstax.org, docs.python.org, pandas.pydata.org, owasp.org, postgresql.org, kaggle.com/learn, mode.com).
2) Choose the closest match for the courseName.
3) Match learning_style:
   - book: prefer openstax.org or official documentation/books within allowlist
   - video: prefer ocw.mit.edu or web.stanford.edu
   - mix: prefer course pages or documentation hubs

Inputs:
- courseName: {courseName}
- learning_style: {learning_style}

If and ONLY IF you cannot choose any suitable URL from the allowlist after multiple attempts:
- set resource to null
- set note explaining why

Return JSON only.
""".strip()

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": GeminiResourcePick.model_json_schema(),
        },
    )
    pick = GeminiResourcePick.model_validate_json(resp.text)

    if pick.resource and not url_is_allowed(pick.resource.url):
        return GeminiResourcePick(resource=None, note="Model suggested non-allowlisted URL, rejected.")
    return pick


def generate_5step_plans(payload: dict) -> RoadmapLiteResponse:
    prompt = f"""
You are an academic planning assistant.

Goal:
For each course, produce a concise Arabic completion plan of EXACTLY 5 steps.

Hard rules:
- Output MUST be valid JSON matching the schema.
- plan_ar must be Arabic.
- Keep courseCode, courseName, resource.title, resource.url exactly as provided (do NOT translate them).
- Do NOT add new resources or links.
- If resource is null, still output 5 steps and mention briefly in note that the resource link is missing.

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": RoadmapLiteResponse.model_json_schema(),
        },
    )
    return RoadmapLiteResponse.model_validate_json(resp.text)


# ── Chat generation ─────────────────────────────────────────────────
def gemini_course_chat(payload: dict) -> str:
    prompt = f"""
أنت مساعد EDVORA.

قواعد صارمة:
- تجيب فقط عن الأسئلة المتعلقة بالمقررات الموجودة في القائمة.
- إذا السؤال غير متعلق بالمقررات، اكتب فقط الجملة التالية بدون أي زيادة:
{IRRELEVANT_MSG_AR}

- اكتب الرد باللغه التي تم بها السؤال لكن اذا كان السؤال مختلط عربي وانجليزي تجيب باللغه العربيه.
- لا تضف أي روابط جديدة أو مصادر جديدة.
- لا تترجم هذه القيم (اكتبها كما هي تمامًا):
courseCode, courseName, resource.title, resource.url

بيانات المقررات (JSON):
{json.dumps(payload["courses"], ensure_ascii=False, indent=2)}

سؤال الطالب:
{payload["question"]}
""".strip()

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return (resp.text or "").strip() or IRRELEVANT_MSG_AR


# ── Routes ──────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"service": "edvora-api", "status": "running"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/catalog/courses", methods=["GET"])
def catalog_courses():
    items = []
    for code, c in sorted(CATALOG.items(), key=lambda x: x[0]):
        items.append({
            "courseCode": c["courseCode"],
            "courseName": c["courseName"],
            "track": c.get("track"),
            "level": c.get("level"),
        })
    return jsonify({"count": len(items), "courses": items})


@app.route("/roadmap-lite", methods=["POST"])
def roadmap_lite():
    try:
        req = RoadmapLiteRequest.model_validate(request.json or {})
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    resolved_courses = []
    for c in req.courses:
        code = c.courseCode.strip()
        name = c.courseName.strip()

        if code in CATALOG:
            cat = CATALOG[code]
            r = pick_resource_from_catalog(cat, req.learning_style)
            resource = None
            note = None
            if r and r.get("url"):
                resource = {"type": r["type"], "title": r["title"], "url": r["url"]}
            else:
                note = "No suitable resource in catalog for this learning_style."
            resolved_courses.append({
                "courseCode": code,
                "courseName": cat["courseName"],
                "skillLevel": c.skillLevel,
                "resource": resource,
                "note": note
            })
        else:
            pick = gemini_pick_resource(courseName=name, learning_style=req.learning_style)
            resolved_courses.append({
                "courseCode": code,
                "courseName": name,
                "skillLevel": c.skillLevel,
                "resource": pick.resource.model_dump() if pick.resource else None,
                "note": pick.note or "Course not in catalog; whitelist-only resource selection attempted."
            })

    model_payload = {
        "learning_style": req.learning_style,
        "weeklyHours": req.weeklyHours,
        "currentLevel": req.currentLevel,
        "academic_goal": req.academic_goal,
        "courses": resolved_courses
    }

    try:
        out = generate_5step_plans(model_payload)
        return jsonify(out.model_dump())
    except Exception as e:
        return jsonify({"error": "model_error", "details": str(e)}), 502


@app.route("/chat", methods=["POST"])
def chat():
    try:
        req = ChatRequest.model_validate(request.json or {})
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    resolved_courses = []
    for c in req.courses:
        code = c.courseCode.strip()
        name = c.courseName.strip()
        if code in CATALOG:
            cat = CATALOG[code]
            r = pick_resource_from_catalog(cat, req.learning_style)
            resource = {"type": r["type"], "title": r["title"], "url": r["url"]} if (r and r.get("url")) else None
            resolved_courses.append({
                "courseCode": code,
                "courseName": cat["courseName"],
                "skillLevel": c.skillLevel,
                "resource": resource
            })
        else:
            pick = gemini_pick_resource(courseName=name, learning_style=req.learning_style)
            resolved_courses.append({
                "courseCode": code,
                "courseName": name,
                "skillLevel": c.skillLevel,
                "resource": pick.resource.model_dump() if pick.resource else None
            })

    payload = {
        "learning_style": req.learning_style,
        "weeklyHours": req.weeklyHours,
        "currentLevel": req.currentLevel,
        "academic_goal": req.academic_goal,
        "courses": resolved_courses,
        "question": req.question
    }

    answer = gemini_course_chat(payload)
    return jsonify({"answer": answer})

"""结构化输出字段、类型和跨字段业务规则。"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StudyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=2, max_length=30)
    difficulty: int = Field(ge=1, le=5)
    keywords: list[str] = Field(min_length=2, max_length=4)
    estimated_minutes: int = Field(ge=10, le=120)

    @field_validator("keywords")
    @classmethod
    def keywords_must_be_unique(cls, value: list[str]) -> list[str]:
        cleaned = [
            item.strip() for item in value if item.strip()
        ]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("keywords 不得重复")
        return cleaned

    @model_validator(mode="after")
    def harder_plan_needs_time(self):
        if self.difficulty >= 4 and self.estimated_minutes < 30:
            raise ValueError("难度达到 4 时，学习时间不得少于 30 分钟")
        return self

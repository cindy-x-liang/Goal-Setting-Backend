from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# instructor_table = db.Table(
#     "instructor",
#     db.Column("course_id", db.Integer, db.ForeignKey("course.id")),
#     db.Column("user_id", db.Integer, db.ForeignKey("user.id"))
# )
# student_table = db.Table(
#     "student",
#     db.Column("course_id", db.Integer, db.ForeignKey("course.id")),
#     db.Column("user_id", db.Integer, db.ForeignKey("user.id"))
# )


class Goals(db.Model):
    """
    goal model
    """

    __tablename__ = "Goal"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String, nullable=False)
    progress = db.Column(db.String, nullable=False)
    description = db.Column(db.String, nullable=False)
    startDate = db.Column(db.String, nullable=False)
    endDate = db.Column(db.String, nullable=False)


    def __init__(self, **kwargs):
        """
        Initializes a User object
        """

        self.title = kwargs.get("title", "")
        self.progress = kwargs.get("progress")
        self.description = kwargs.get("description")
        self.startDate = kwargs.get("startDate")
        self.endDate = kwargs.get("endDate")

    def to_dict(self):
        """
        Serializes a user object
        """
        return {
            "id": self.id,
            "title": self.title,
            "progress": self.progress,
            "description": self.description,
            "startDate": self.startDate,
            "endDate": self.endDate
        }

    def simple_serialize(self):
        """
        Serialize a User object without the courses field
        """

        return {
            "id": self.id,
            "title": self.title,
            "progress": self.progress,
            "description": self.description,
            "startDate": self.startDate,
            "endDate": self.endDate
        }

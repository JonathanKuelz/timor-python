// Main file for ik calculation with C++ and pybind11 using pinocchio

#include <math.h>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>  // Convert numpy <-> Eigen
#include <pybind11/stl.h>  // Convert python list <-> std::list
#include <pinocchio/bindings/python/pybind11.hpp>  // Convert pinocchio <-> numpy
#define SCALAR double
#define OPTIONS 0
#define JOINT_MODEL_COLLECTION ::pinocchio::JointCollectionDefaultTpl

#include <pinocchio/bindings/python/pybind11-all.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <pinocchio/algorithm/jacobian.hpp> // compute Jacobian

#include <pinocchio/algorithm/geometry.hpp>

namespace py = pybind11;
namespace pin = pinocchio;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

pin::SE3 fk(pin::Model& model, pin::Data& data, Eigen::VectorXd q, int frame) {
    /*
    Implement a pinocchio fk method to test that interface to python side works.
    */
    pin::forwardKinematics(model, data, q);
    pin::updateFramePlacement(model, data, frame);

    pin::SE3 pose = data.oMf[frame];

    return pose;
}

bool has_self_collision(pin::Model& model, pin::Data& data, const pin::GeometryModel& geom_model,
                        Eigen::VectorXd q, pin::GeometryData& geom_data) {
    /*
    Implement a method to check for self collision of robot.
    */
    return pin::computeCollisions(model, data, geom_model, geom_data, q, true);
}

bool has_self_collision(pin::Model& model, pin::Data& data, const pin::GeometryModel& geom_model,
                        Eigen::VectorXd q) {
    /*
    Implement a method to check for self collision of robot.
    */
    pin::GeometryData geom_data(geom_model);  // Pretty expensive; don't do to often
    return pin::computeCollisions(model, data, geom_model, geom_data, q, true);
}

ArrayXb has_self_collision_vec(pin::Model& model, pin::Data& data, const pin::GeometryModel& geom_model,
                               Eigen::MatrixXd q) {
    /*
    Check multiple configurations for self collision.
    */
    pin::GeometryData geom_data(geom_model);  // Pretty expensive; don't do to often
    assert (q.cols() == model.nq);
    ArrayXb collision(q.rows());
    for (int i=0; i<q.rows(); i++) {
        // collision[i] = pin::computeCollisions(model, data, geom_model, geom_data, q.row(i), true);
        collision[i] = has_self_collision(model, data, geom_model, q.row(i), geom_data);
    }
    return collision;
}

double ik_default_cost_function(pin::SE3 desired, Eigen::VectorXd q,
                                pin::Model& model, pin::Data& data, int frame,
                                double weight_translation=1., double weight_rotation=.5/M_PI) {
    /*
    Implement a default cost function for ik; same as timor.Robot.default_ik_cost_function
    */
    pin::forwardKinematics(model, data, q);
    pin::updateFramePlacement(model, data, frame);  // make sure oMf actually calculated
    const pin::SE3 iMd = data.oMf[frame].actInv(desired);
    double transl_error = iMd.translation().norm();
    double rotation_error = acos((iMd.rotation().trace() - 1) / 2);  // Rotation angle of rotation matrix
    return (weight_translation * transl_error + weight_rotation * rotation_error) /
        (weight_translation + weight_rotation);
}

std::tuple<Eigen::VectorXd, bool, int> ik(pin::Model& model, pin::Data& data, pin::SE3 desired, int frame,
                                          Eigen::VectorXd q_init,
                                          const pin::GeometryModel& geom_model = pin::GeometryModel(),
                                          double eps=1e-4, int max_iter=1000,
                                          double DT=1e-1, double damp=1e-6, bool random_restart=true,
                                          double alpha_average=0.5, double convergence_threshold=1e-8,
                                          double weight_translation=1., double weight_rotation=.5/M_PI) {
    /*
    Implement a jacobian based ik
    */
    Eigen::VectorXd q = q_init;  // Accumulation of result set to initial guess
    Eigen::VectorXd q_best = q;  // Best result so far

    // Create geometry data one as it is expensive
    pin::GeometryData geom_data(geom_model);

    pin::Data::Matrix6x J(6,model.nv);
    J.setZero();

    bool success = false;  // Flag to indicate if we have reached the desired pose satisfying all constraints tested
    bool restart = false;  // Flag to indicate if we should restart with random configuration
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d err;
    Eigen::VectorXd v(model.nv);
    double previous_cost = ik_default_cost_function(desired, q, model, data, frame,
                                                    weight_translation, weight_rotation);
    double best_cost = previous_cost;
    double mean_improvement = 0;
    int i=max_iter;
    for (; i>=0; i--)
    {
        if (restart) {
            if (!random_restart) {
                break;
            }
            pin::randomConfiguration(model, model.lowerPositionLimit, model.upperPositionLimit, q);
            restart = false;
        }
        pin::forwardKinematics(model,data,q);
        pin::updateFramePlacement(model, data, frame);  // make sure oMf actually calculated
        const pin::SE3 iMd = data.oMf[frame].actInv(desired);
        err = pin::log6(iMd).toVector();  // in joint frame
        // Success condition - within tolerance and within joint limits
        if (err.norm() < eps) {
            // Check if outside joint limits -> restart
            if ((q.array() < model.lowerPositionLimit.array()).any() ||
                (q.array() > model.upperPositionLimit.array()).any()) {
                restart = true;
                continue;
            }
            // Check if self collision -> restart
            if (has_self_collision(model, data, geom_model, q, geom_data)) {
                restart = true;
                continue;
            }
            success = true;
            q_best = q;
            break;
        }
        // Take a step
        pin::computeFrameJacobian(model, data, q, frame, J);  // J in joint frame
        pin::Data::Matrix6 Jlog;
        pin::Jlog6(iMd.inverse(), Jlog);
        J = -Jlog * J;
        pin::Data::Matrix6 JJt;
        JJt.noalias() = J * J.transpose();  // Add other inversion methods here
        JJt.diagonal().array() += damp;
        v.noalias() = - J.transpose() * JJt.ldlt().solve(err);
        q = pin::integrate(model, q, v*DT);
        // Add wrap to joint limits / locking joints here
        // Check for convergence without arriving at the desired pose
        double next_cost = ik_default_cost_function(desired, q, model, data, frame);
        double improvement = previous_cost - next_cost;
        mean_improvement = mean_improvement * alpha_average + improvement * (1 - alpha_average);
        if (mean_improvement < convergence_threshold) {
            restart = true;
        }
        previous_cost = next_cost;
        if (next_cost < best_cost) {
            q_best = q;
            best_cost = next_cost;
        }
    }
    // Restart if outside joint limits
    return std::make_tuple(q_best, success, i);
}

PYBIND11_MODULE(ik_cpp, m) {
    m.doc() = R"pbdoc(
    Pybind11 of C++ IK solver using pinocchio.
    )pbdoc";

    m.def("fk", pin::python::make_pybind11_function(&fk), R"pbdoc(
        Get forward kinematics of trajectory at time within trajectory.

        :param model: pinocchio model of robot.
        :param data: pinocchio data of robot.
        :param q: joint configuration vector.
    )pbdoc");

    m.def("has_self_collision",
          pin::python::make_pybind11_function(
              py::overload_cast<pin::Model&, pin::Data&, const pin::GeometryModel&, Eigen::VectorXd>(
                  &has_self_collision)
          ),
          R"pbdoc(
          Check if robot has self collision.

          :param model: pinocchio model of robot.
          :param data: pinocchio data of robot.
          :param geom_model: pinocchio geometry model of robot.
          :param q: joint configuration vector.
          )pbdoc",
          py::arg("model"), py::arg("data"), py::arg("geom_model"), py::arg("q"));

    m.def("has_self_collision_vec",
          pin::python::make_pybind11_function(&has_self_collision_vec),
          R"pbdoc(
          Vectorized version to check if robot has self collision.

          :param model: pinocchio model of robot.
          :param data: pinocchio data of robot.
          :param geom_model: pinocchio geometry model of robot.
          :param q: joint configuration vector (n_samples x nDoF).
          )pbdoc",
          py::arg("model"), py::arg("data"), py::arg("geom_model"), py::arg("q"));

    m.def("ik", pin::python::make_pybind11_function(&ik), R"pbdoc(
            Get inverse kinematics of trajectory at time within trajectory.

            :param model: pinocchio model of robot.
            :param data: pinocchio data of robot.
            :param desired: desired eef pose.
            :param frame: frame id of eef within model and data.
            :param q_init: initial guess of joint configuration.
            :param geom_model: pinocchio geometry model of robot used for self-collision avoidance;
              please provide empty model if not needed (i.e., `pin.GeometryModel()`) as pybind11 struggles with
              `pin::GeometryModel()` as a default value on the C++ side.
            :param eps: tolerance of error (norm of log map between is and desired eef pose).
            :param max_iter: maximum number of iterations.
            :param DT: time step size for integration.
            :param damp: damping factor.
            :param random_restart: whether to restart with random configuration.
            :param alpha_average: averaging factor for convergence.
            :param convergence_threshold: threshold for convergence.
            :param weight_translation: weight for translation error in cost function.
            :param weight_rotation: weight for rotation error in cost function.
            :returns: (q, success, iter_remaining) joint configuration, success flag, remaining iterations.
        )pbdoc",
        py::arg("model"), py::arg("data"), py::arg("desired"), py::arg("frame"), py::arg("q_init"),
        py::arg("geom_model"), py::arg("eps") = 1e-4, py::arg("max_iter") = 1000, py::arg("DT") = 1e-1,
        py::arg("damp") = 1e-6, py::arg("random_restart") = false, py::arg("alpha_average") = 0.5,
        py::arg("convergence_threshold") = 1e-8,
        py::arg("weight_translation") = 1., py::arg("weight_rotation") = .5/M_PI);

    m.attr("__version__") = "dev";
}

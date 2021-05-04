#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/constrained_linear_operator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

template<int dim>
class MagneticField
{
    public:
        MagneticField() {}

        Tensor<1, dim> get_force(const Tensor<1,dim> &p) const
        {
            (void)p;
            Tensor<1, dim> val;
            val[0] = 0.0;
            val[1] = 0.0;
            return val;
        }

};

template<int dim>
class ExactVelocity : public Function<dim>
{
    public:
        ExactVelocity()
            : Function<dim>(dim) // 3
        {}

        virtual void
        vector_value(const Point<dim> &p,
                     Vector<double> &  values) const override
        {
            const double time = this->get_time();
            //const double Um = 1.5;
            //const double H  = 4.1;
            //values(0) = 4. * Um * p(1) * (H - p(1)) / (H * H);
            //values(1) = 0.0;
            values(0) = time*cos(p(1));
            values(1) = time*sin(p(0));
        }
};

template<int dim>
class ExactPressure : public Function<dim>
{
    public:
        ExactPressure()
            : Function<dim>(1) // 3
        {}

        virtual 
        double value(const Point<dim> &p, unsigned int component = 0) const override
        {
            (void)component;
            const double time = this->get_time();
            //return 0.0;
            return time*p(0)*p(1);
            //return 25.0 - p(0);
        }
};


template <int dim>
class NavierStokes
{
    public:
        NavierStokes(const unsigned int degree, 
                const unsigned int n_global_refinements);


        void run();

    private:
        void setup_grid();
        void setup_dofs();
        void assemble_matrices();
        void initialize_system();
        void diffusion_step();
        void projection_step();
        void pressure_step();
        void assemble_advection_matrix();
        void output_results();

        const unsigned int degree;
        unsigned int n_global_refinements;

        double dt;
        double end_time;
        double current_time;
        int    time_step_number;

        double eta;
        double mu;
        double rho;

        bool standard;
        MagneticField<dim> mag_field;

        Triangulation<dim> triangulation;
        FESystem<dim>      fe_u;
        FE_Q<dim>          fe_p;
        DoFHandler<dim>    dof_handler_u;
        DoFHandler<dim>    dof_handler_p;

        AffineConstraints<double> u_constraints;
        AffineConstraints<double> p_constraints;

        SparsityPattern      sparsity_pattern_u;
        SparsityPattern      sparsity_pattern_p;
        SparsityPattern      sparsity_pattern_g;

        SparseMatrix<double> laplace_matrix_u;
        SparseMatrix<double> laplace_matrix_p;

        SparseMatrix<double> mass_matrix_u;
        SparseMatrix<double> mass_matrix_p;

        SparseMatrix<double> advection_matrix;
        SparseMatrix<double> gradient_matrix;

        SparseMatrix<double> velocity_step_matrix_const;
        SparseMatrix<double> velocity_step_matrix;
        SparseMatrix<double> pressure_step_matrix;

        Vector<double> diffusion_rhs;
        Vector<double> projection_rhs;

        Vector<double> prev_u;
        Vector<double> curr_u;
        Vector<double> prev_p;
        Vector<double> curr_p;
        Vector<double> prev_phi;
        Vector<double> curr_phi;

        Vector<double> u_star;
        Vector<double> p_star;

        SparseDirectUMFPACK pressure_mass_inv;
        SparseDirectUMFPACK pressure_laplace_inv;
        SparseILU<double>   diffusion_preconditioner;
        SparseILU<double>   projection_preconditioner;

        ExactVelocity<dim> exact_u;
        ExactPressure<dim> exact_p;
};

template <int dim>
class ForcingFunction : public Function<dim>
{
    public:
        ForcingFunction(double eta, double mu, MagneticField<dim> &m_field)
            : Function<dim>(dim)
            , eta(eta)
            , mu(mu)
            , m_field(m_field)// 3
        {}

        virtual void
        vector_value(const Point<dim> &p,
                     Vector<double> &  values) const override
        {
            const double time = this->get_time();
            //(void)p;
            values(0) = mu*time*cos(p(1)) - time*time*sin(p(0))*sin(p(1)) +
                        time*p(1) + cos(p(1));
            values(1) = mu*time*sin(p(0)) + time*time*cos(p(0))*cos(p(1)) +
                        time*p(0) + sin(p(0));
        }
    private:
        double eta;
        double mu;
        MagneticField<dim> &m_field;
};

template <int dim>
NavierStokes<dim>::NavierStokes(const unsigned int degree,
                                const unsigned int n_global_refinements)
    : degree(degree)
    , n_global_refinements(n_global_refinements)
    , dt(std::pow(0.1, n_global_refinements))
    , current_time(0.0)
    , time_step_number(3)
    , eta(0.1)
    , mu(1.0)
    , rho(1.0)
    , standard(false)
    , fe_u(FE_Q<dim>(degree+1), dim)
    , fe_p(degree)
    , dof_handler_u(triangulation)
    , dof_handler_p(triangulation)
{}

template <int dim>
void
NavierStokes<dim>::setup_grid()
{
  GridGenerator::hyper_cube(triangulation);

  //std::vector<GridTools::PeriodicFacePair<
  //    typename parallel::distributed::Triangulation<dim>::cell_iterator>>
  //    periodicity_vector;
  //GridTools::collect_periodic_faces(triangulation,
  //                                   2,
  //                                    3,
  //                                    1,
  //                                    periodicity_vector,
  //                                    Tensor<1, dim>(),
  //                                    rotation_matrix);

  triangulation.refine_global(n_global_refinements);
  double dx = GridTools::minimal_cell_diameter(triangulation);
  dt = dx;
}


template <int dim>
void NavierStokes<dim>::setup_dofs()
{
    laplace_matrix_u.clear();
    laplace_matrix_p.clear();
    mass_matrix_u.clear();
    mass_matrix_p.clear();
    advection_matrix.clear();
    gradient_matrix.clear();

    dof_handler_u.distribute_dofs(fe_u);
    DoFRenumbering::Cuthill_McKee(dof_handler_u);

    dof_handler_p.distribute_dofs(fe_p);
    DoFRenumbering::Cuthill_McKee(dof_handler_p);

    prev_u.reinit(dof_handler_u.n_dofs());
    prev_p.reinit(dof_handler_p.n_dofs());
    prev_phi.reinit(dof_handler_p.n_dofs());
    
    curr_u.reinit(dof_handler_u.n_dofs());
    curr_p.reinit(dof_handler_p.n_dofs());
    curr_phi.reinit(dof_handler_p.n_dofs());

    u_star.reinit(dof_handler_u.n_dofs());
    p_star.reinit(dof_handler_p.n_dofs());

    diffusion_rhs.reinit(dof_handler_u.n_dofs());
    projection_rhs.reinit(dof_handler_p.n_dofs());

    // overkill to use exact solution here
    exact_u.set_time(current_time);
    exact_p.set_time(current_time);

    VectorTools::interpolate_boundary_values(
            dof_handler_u, 0, exact_u, u_constraints);
    u_constraints.close();

    VectorTools::interpolate_boundary_values(
            dof_handler_p, 0, exact_p, p_constraints);
    p_constraints.close();
}

template <int dim>
void NavierStokes<dim>::assemble_matrices()
{
    QGauss<dim> qf(degree+2);

    // velocity sparsity pattern
    DynamicSparsityPattern u_dsp(dof_handler_u.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_u,
                                    u_dsp,
                                    u_constraints,
                                    /* keep_constrained_dofs */ true);
    sparsity_pattern_u.copy_from(u_dsp);

    // initialize velocity matrices
    laplace_matrix_u.reinit(sparsity_pattern_u);
    mass_matrix_u.reinit(sparsity_pattern_u);
    advection_matrix.reinit(sparsity_pattern_u);
    velocity_step_matrix_const.reinit(sparsity_pattern_u);
    velocity_step_matrix.reinit(sparsity_pattern_u);

    MatrixCreator::create_mass_matrix(dof_handler_u,
                                      qf,
                                      mass_matrix_u);
    MatrixCreator::create_laplace_matrix(dof_handler_u,
                                         qf,
                                         laplace_matrix_u);
    velocity_step_matrix_const = 0.0;
    velocity_step_matrix_const.add(mu, laplace_matrix_u);
    velocity_step_matrix_const.add(1.5/dt, mass_matrix_u);

    // pressure sparsity pattern
    DynamicSparsityPattern p_dsp(dof_handler_p.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_p,
                                    p_dsp,
                                    p_constraints,
                                    /* keep_constrained_dofs */ true);
    sparsity_pattern_p.copy_from(p_dsp);

    // initialize pressure matrices
    laplace_matrix_p.reinit(sparsity_pattern_p);
    mass_matrix_p.reinit(sparsity_pattern_p);
    pressure_step_matrix.reinit(sparsity_pattern_p);

    MatrixCreator::create_mass_matrix(dof_handler_p,
                                      qf,
                                      mass_matrix_p);
    MatrixCreator::create_laplace_matrix(dof_handler_p,
                                         qf,
                                         laplace_matrix_p);

    // velocity-pressure sparsity pattern
    DynamicSparsityPattern g_dsp(dof_handler_u.n_dofs(),
                                 dof_handler_p.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_u,
                                    dof_handler_p,
                                    g_dsp);
    sparsity_pattern_g.copy_from(g_dsp);

    // initialize gradient operator matrix
    gradient_matrix.reinit(sparsity_pattern_g);

    FEValues<dim> fe_values_u(fe_u, qf, 
                              update_gradients | update_JxW_values);
    FEValues<dim> fe_values_p(fe_p, qf, 
                              update_values | update_JxW_values);

    FEValuesViews::Vector<dim> fe_views_u(fe_values_u, 0);

    const unsigned int dofs_per_cell_u = fe_u.n_dofs_per_cell();
    const unsigned int dofs_per_cell_p = fe_p.n_dofs_per_cell();
    const unsigned int n_q_points = qf.size();

    FullMatrix<double> cell_gradient_matrix(dofs_per_cell_u, dofs_per_cell_p);
    std::vector<types::global_dof_index> local_dof_indices_u(dofs_per_cell_u);
    std::vector<types::global_dof_index> local_dof_indices_p(dofs_per_cell_p);

    double div_i, val_j;

    auto cell_u = dof_handler_u.begin_active();
    const auto cell_end_u = dof_handler_u.end();
    auto cell_p = dof_handler_p.begin_active();
    while (cell_u != cell_end_u)
    {
      Assert(cell_u->center() == cell_p->center(), 
             ExcMessage("a real bad thing happened"));

      cell_gradient_matrix = 0;

      fe_values_u.reinit(cell_u);
      fe_values_p.reinit(cell_p);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell_u; ++i)
            {
              div_i = fe_views_u.divergence(i, q_index);

              for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
                {
                  val_j = fe_values_p.shape_value(j, q_index);

                  cell_gradient_matrix(i,j) += 
                      -fe_values_u.JxW(q_index) * div_i * val_j;
                }
            }
        }
      cell_u->get_dof_indices(local_dof_indices_u);
      cell_p->get_dof_indices(local_dof_indices_p);
      for (unsigned int i = 0; i < dofs_per_cell_u; ++i)
          for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
              gradient_matrix.add(local_dof_indices_u[i],
                                  local_dof_indices_p[j],
                                  cell_gradient_matrix(i,j));

      ++cell_u;
      ++cell_p;
    }
}

template<int dim>
void NavierStokes<dim>::initialize_system()
{

    // use exact solution to set up initial condition
    VectorTools::interpolate(dof_handler_u,
                             exact_u,
                             prev_u);
    exact_u.advance_time(dt);
    VectorTools::interpolate(dof_handler_u,
                             exact_u,
                             curr_u);

    VectorTools::interpolate(dof_handler_p,
                             exact_p,
                             prev_p);
    exact_p.advance_time(dt);
    VectorTools::interpolate(dof_handler_p,
                             exact_p,
                             curr_p);
    prev_phi = 0.0;
    curr_phi = 0.0;
    curr_phi.add(1.0, curr_p, -1.0, prev_p); // set phi_1 = p_1 - p_0

    // 
    projection_preconditioner.initialize(laplace_matrix_p);
    pressure_mass_inv.factorize(mass_matrix_p);
}

template <int dim>
void NavierStokes<dim>::assemble_advection_matrix()
{
    advection_matrix = 0;

    QGauss<dim> quadrature_formula(degree+2);

    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping,
                            fe_u,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);

    FEValuesViews::Vector<dim> fe_views(fe_values, 0);

    const unsigned int dofs_per_cell = fe_u.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double>     cell_advection_matrix(dofs_per_cell, dofs_per_cell);

    //std::vector<Tensor<1,dim>> u_star_values(n_q_points,
    //                                         Tensor<1,dim,double>);
    //std::vector<Tensor<1,dim>> u_star_gradients(n_uq_points);
    std::vector<Tensor<1,dim>> u_star_values(n_q_points);
    std::vector<Tensor<2,dim>> u_star_gradients(n_q_points);

    Tensor<1, dim>              val_i_u, val_j_u;
    Tensor<2, dim>              grad_j_u;
    double                      u_star_div;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    for (const auto &cell : dof_handler_u.active_cell_iterators())
    {
      cell_advection_matrix = 0;

      fe_values.reinit(cell);

      fe_views.get_function_values(u_star, u_star_values);
      fe_views.get_function_gradients(u_star, u_star_gradients);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
          u_star_div = 0;

          for ( unsigned int d = 0; d < dim; ++d)
              u_star_div += u_star_gradients[q_index][d][d];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              val_i_u = fe_views.value(i, q_index);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  val_j_u = fe_views.value(j, q_index);
                  grad_j_u = fe_views.gradient(j, q_index);

                  cell_advection_matrix(i,j) +=
                      val_i_u * 
                      ( u_star_values[q_index] * grad_j_u + 
                        0.5 * u_star_div * val_j_u ) *
                      fe_values.JxW(q_index);
                }
            }
        }

      cell->get_dof_indices(local_dof_indices);

      u_constraints.distribute_local_to_global(
        cell_advection_matrix, local_dof_indices, advection_matrix); 

    }
}

template<int dim>
void NavierStokes<dim>::diffusion_step()
{
    // define u *
    u_star.equ(2.0, curr_u);
    u_star -= prev_u;

    // define p #
    p_star.equ(-1.0, curr_p);
    p_star.add(-4.0/3.0, curr_phi, 1.0/3.0, prev_phi);
    //p_star *= -1.0;

    assemble_advection_matrix();

    // define (v, f^k+1)
    MappingQGeneric<dim> mapping(1);
    ForcingFunction<dim> forcing_function(eta, mu, mag_field);
    forcing_function.set_time(current_time + dt);
    VectorTools::create_right_hand_side(mapping,
                                        dof_handler_u,
                                        QGauss<dim>(degree + 2),
                                        forcing_function,
                                        diffusion_rhs);

    // add 1/2dt (v, 4uk - uk-1) 
    Vector<double> tmp(dof_handler_u.n_dofs());
    tmp.equ(2.0/dt, curr_u);
    tmp.add(-0.5/dt, prev_u);
    mass_matrix_u.vmult_add(diffusion_rhs, tmp);

    // add (div v, p #)
    gradient_matrix.vmult_add(diffusion_rhs, p_star);

    // set u_n-1 = u_n
    //std::swap(prev_u, curr_u);
    prev_u = curr_u;

    // define 3/2dt M + A + nu K
    velocity_step_matrix.copy_from(velocity_step_matrix_const);
    velocity_step_matrix.add(1.0, advection_matrix);


    // boundary conditions
    exact_u.set_time(current_time+dt);
    std::map<types::global_dof_index, double> boundary_values_u;
    VectorTools::interpolate_boundary_values(dof_handler_u,
                                             0,
                                             exact_u,
                                             boundary_values_u);
    MatrixTools::apply_boundary_values(boundary_values_u,
                                       velocity_step_matrix,
                                       curr_u,
                                       diffusion_rhs);

    // initialize preconditioner
    diffusion_preconditioner.initialize(velocity_step_matrix);

    // solver
    SolverControl solver_control(1000, 1e-8*diffusion_rhs.l2_norm());
    SolverGMRES<Vector<double>> gmres(solver_control);
    gmres.solve(velocity_step_matrix, curr_u, 
                diffusion_rhs, diffusion_preconditioner);

    //VectorTools::interpolate(dof_handler_u,
    //                         exact_u,
    //                         curr_u);

}


template<int dim>
void NavierStokes<dim>::projection_step()
{
    // get laplace matrix
    pressure_step_matrix.copy_from(laplace_matrix_p);

    projection_rhs = 0.0;
    // define projection right hand side
    gradient_matrix.Tvmult_add(projection_rhs, curr_u);

    // iterate phi
    //std::swap(prev_phi, curr_phi);
    prev_phi = curr_phi;

    // boundary condition.
    static std::map<types::global_dof_index, double> boundary_values_phi;
    
    if (time_step_number == 3)
        VectorTools::interpolate_boundary_values(dof_handler_p,
                                                 0,
                                                 Functions::ZeroFunction<dim>(),
                                                 boundary_values_phi);
    MatrixTools::apply_boundary_values(boundary_values_phi,
                                       pressure_step_matrix,
                                       curr_phi,
                                       projection_rhs);

    // solver
    SolverControl solver_control(1000, 1e-8*projection_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    cg.solve(pressure_step_matrix, curr_phi, 
             projection_rhs, projection_preconditioner);
    curr_phi *= 1.5/dt;
}


template<int dim>
void NavierStokes<dim>::pressure_step()
{

    // set p_n-1 = p_n
    //std::swap(prev_p, curr_p);
    prev_p = curr_p;

    if (standard)
        curr_p += curr_phi;
    else
    {
        curr_p = projection_rhs;
        pressure_mass_inv.solve(curr_p);
        curr_p.sadd(1.0*mu, 1.0, prev_p);
        curr_p += curr_phi;
    }

}


template <int dim>
void NavierStokes<dim>::run()
{
    setup_grid();
    setup_dofs();

    std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells() << std::endl;
    std::cout << "   Number of degrees of freedom: " << dof_handler_u.n_dofs() << " " <<
         dof_handler_p.n_dofs()
            << '\n'
            << "   Timestep size: "
            << dt
            << std::endl;

    assemble_matrices();
    initialize_system();

    for (; current_time <= 0.5; current_time += dt, ++time_step_number)
    {
        exact_u.advance_time(dt);

        diffusion_step();
        projection_step();
        pressure_step();
    }


    std::cout << std::endl;

  output_results();

}

template <int dim>
void NavierStokes<dim>::output_results() 
{
    // Compute the pointwise maximum error:
    Vector<double> max_error_per_cell_u(triangulation.n_active_cells());
    {
        exact_u.set_time(current_time);

        MappingQGeneric<dim> mapping(1);
        VectorTools::integrate_difference(mapping,
                                          dof_handler_u,
                                          curr_u,
                                          exact_u,
                                          max_error_per_cell_u,
                                          QIterated<dim>(QGauss<1>(2), 2),
                                          VectorTools::NormType::L2_norm);
                                          //VectorTools::NormType::Linfty_norm);
        std::cout << "\t\t\t\tmaximum u error = "
                  << *std::max_element(max_error_per_cell_u.begin(),
                                       max_error_per_cell_u.end())
                  << std::endl;
    }
    Vector<double> max_error_per_cell_p(triangulation.n_active_cells());
    {
        exact_p.set_time(current_time);

        MappingQGeneric<dim> mapping(1);
        VectorTools::integrate_difference(mapping,
                                          dof_handler_p,
                                          curr_p,
                                          exact_p,
                                          max_error_per_cell_p,
                                          QIterated<dim>(QGauss<1>(2), 2),
                                          VectorTools::NormType::L2_norm);
                                          //VectorTools::NormType::Linfty_norm);
        std::cout << "\t\t\t\tmaximum p error = "
                  << *std::max_element(max_error_per_cell_p.begin(),
                                       max_error_per_cell_p.end())
                  << std::endl;
    }

    // join the two systems
    const FESystem<dim> joint_fe(fe_u, 1, fe_p, 1);
    DoFHandler<dim> joint_dof_handler(triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);
    Assert(joint_dof_handler.n_dofs() ==
           ( dof_handler_u.n_dofs() +
            dof_handler_p.n_dofs()),
            ExcInternalError());

    Vector<double> joint_solution(joint_dof_handler.n_dofs());
    std::vector<types::global_dof_index> 
      loc_joint_dof_indices(joint_fe.n_dofs_per_cell()),
      loc_vel_dof_indices(fe_u.n_dofs_per_cell()),
      loc_pres_dof_indices(fe_p.n_dofs_per_cell());

    typename DoFHandler<dim>::active_cell_iterator
      joint_cell = joint_dof_handler.begin_active(),
      joint_endc = joint_dof_handler.end(),
      vel_cell   = dof_handler_u.begin_active(),
      pres_cell  = dof_handler_p.begin_active();
    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell, ++pres_cell)
      {
        joint_cell->get_dof_indices(loc_joint_dof_indices);
        vel_cell->get_dof_indices(loc_vel_dof_indices);
        pres_cell->get_dof_indices(loc_pres_dof_indices);
        for (unsigned int i = 0; i < joint_fe.n_dofs_per_cell(); ++i)
            if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                    joint_solution(loc_joint_dof_indices[i]) = curr_u(
                        loc_vel_dof_indices[joint_fe.system_to_base_index(i)
                                          .second]);
                }
              else
                {
                    joint_solution(loc_joint_dof_indices[i]) = curr_p(
                        loc_pres_dof_indices[joint_fe.system_to_base_index(i)
                                          .second]);
                }
      }
                  

        /*
        switch (joint_fe.system_to_base_index(i).first.first)
          {
            case 0:
              //Assert(joint_fe.system_to_base_index(i).first.second < dim,
              //       ExcInternalError());
              joint_solution(loc_joint_dof_indices[i]) = curr_u(
                  loc_vel_dof_indices[joint_fe.system_to_base_index(i)
                                        .second]);
              break;
            case 1:
              Assert(joint_fe.system_to_base_index(i).first.second == 0,
                     ExcInternalError());
              joint_solution(loc_joint_dof_indices[i]) = curr_p(
                  loc_pres_dof_indices[joint_fe.system_to_base_index(i)
                                              .second]);
              break;
            default:
              Assert(false, ExcInternalError());
          }
          */
    std::vector<std::string> joint_solution_names(dim, "velocity");
    joint_solution_names.emplace_back("pressure");

    // Save the output:
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(joint_dof_handler);

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          component_interpretation(
            dim + 1, DataComponentInterpretation::component_is_part_of_vector);
        component_interpretation[dim] =
          DataComponentInterpretation::component_is_scalar;

        data_out.add_data_vector(joint_solution,
                                 joint_solution_names,
                                 DataOut<dim>::type_dof_data,
                                 component_interpretation);
        data_out.add_data_vector(max_error_per_cell_u, "max_error_per_cell_u");
        data_out.add_data_vector(max_error_per_cell_p, "max_error_per_cell_p");
        data_out.build_patches();

        std::ofstream output("output-" + std::to_string(dim) + "d-" +
                             std::to_string(n_global_refinements) + ".vtu");
        data_out.write_vtu(output);
    }
}


int
main()
{
  int degree = 1;

  for (unsigned int i = 0; i < 6; ++i)
    {
      NavierStokes<2> ns(degree, i);
      ns.run();
    }

  return 0;
}

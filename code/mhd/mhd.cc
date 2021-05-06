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
#include <deal.II/fe/fe_nedelec_sz.h>
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

// Inverse Matrix Class used for creating the pressure mass matrix
// preconditioner (for the outer solve) and used to define the inverse
// of the Schur complement -- both for the Maxwell solver
template<class MatrixType, class PreconditionerType>
class InverseMatrix : public Subscriptor
{
    public:
        InverseMatrix(const MatrixType & m,
                      const PreconditionerType &p)
            : matrix(&m)
            , preconditioner(&p)
        {}

        void vmult(Vector<double> &sol, const Vector<double> &vec) const
        {
            SolverControl solver_control(vec.size(), 1e-6*vec.l2_norm());
            SolverGMRES<Vector<double>> gmres(solver_control);
            sol = 0;
            gmres.solve(*matrix, sol, vec, *preconditioner);
        }

    private:
        const SmartPointer<const MatrixType> matrix;
        const SmartPointer<const PreconditionerType> preconditioner;
};

// Defines the Schur complement given the block system and the A inverse
// used in Maxwell Solver
template<class InverseType>
class SchurComplement : public Subscriptor
{
    public:
        SchurComplement(const BlockSparseMatrix<double> &system_matrix,
                        const InverseType &A_inv)
            : system_matrix(&system_matrix)
            , A_inv(&A_inv)
            , tmp1(system_matrix.block(0,0).m())
            , tmp2(system_matrix.block(0,0).m())
        {}

        void vmult(Vector<double> &sol, const Vector<double> &vec) const
        {
            system_matrix->block(0,1).vmult(tmp1, vec);
            A_inv->vmult(tmp2, tmp1);
            system_matrix->block(1,0).vmult(sol, tmp2);
        }

    private:
        const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
        const SmartPointer<const InverseType> A_inv;

        mutable Vector<double> tmp1, tmp2;
};


// Exact solution for the magnetic field and 
// for the lagrange multiplier in maxwell system
template<int dim>
class ExactMaxwell : public Function<dim>
{
    public:
        ExactMaxwell()
            : Function<dim>(dim+1) // 3
        {}

        virtual void
        vector_value(const Point<dim> &p,
                     Vector<double> &  values) const override
        {
            //const double time = this->get_time();
            (void)p;
            values(0) = 1.0;//time*cos(p(1));
            values(1) = 1.0;//time*sin(p(0));
            values(2) = 0;
        }
};

// Exact solution for velocity
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
            //const double time = this->get_time();
            (void)p;
            values(0) = 1.0;
            values(1) = 1.0;
        }
};

// Exact solution for pressure
template<int dim>
class ExactPressure : public Function<dim>
{
    public:
        ExactPressure()
            : Function<dim>(1) // 3
        {}

        virtual
        double value(const Point<dim> &p, 
                     unsigned int component = 0) const override
        {
            (void)p;
            (void)component;
            //const double time = this->get_time();
            return 1.0;
        }
};

// Main class for MHD solver
template <int dim>
class MHD
{
    public:
        MHD(const unsigned int degree,
            const unsigned int n_global_refinements);

        void run();

    private:
        void setup_grid();
        void output_results();

        // maxwell functions
        void setup_maxwell_dofs();
        void assemble_maxwell_system(bool initial);
        void solve_maxwell(bool update);

        // navier stokes functions
        void setup_ns_dofs();
        void assemble_ns_matrices();
        void initialize_ns_system();
        void assemble_lorenz_force();
        void ns_diffusion_step();
        void ns_projection_step();
        void ns_pressure_step();
        void assemble_ns_advection_matrix();
        void solve_ns();

        const unsigned int degree;
        unsigned int n_global_refinements;

        // time stepping
        double time_step;
        double end_time;
        double current_time;
        int    time_step_number;

        // parameters
        double nu;
        double beta;
        double eta;
        double mu;
        double rho;

        // which pressure update
        bool standard;

        Triangulation<dim> triangulation;

        // finite elements
        FESystem<dim>      fe_maxwell;
        FESystem<dim>      fe_velocity;
        FE_Q<dim>          fe_pressure;

        // dof handlers
        DoFHandler<dim>    maxwell_dof_handler;
        DoFHandler<dim>    velocity_dof_handler;
        DoFHandler<dim>    pressure_dof_handler;

        // constraints
        AffineConstraints<double> maxwell_constraints;
        AffineConstraints<double> velocity_constraints;
        AffineConstraints<double> pressure_constraints;

        // matrices for maxwell solver
        BlockSparsityPattern      maxwell_sparsity_pattern;
        BlockSparsityPattern      maxwell_unconstrained_sparsity_pattern;
        BlockSparseMatrix<double> maxwell_system_matrix;

        BlockSparseMatrix<double> maxwell_unconstrained_system_matrix;
        BlockSparseMatrix<double> maxwell_unconstrained_mass_matrix;

        BlockSparsityPattern      maxwell_preconditioner_sparsity_pattern;
        BlockSparseMatrix<double> maxwell_preconditioner_matrix;
        SparseMatrix<double>      maxwell_lagrange_mass_matrix;

        // vectors for maxwell solver
        BlockVector<double> current_maxwell;
        BlockVector<double> previous_maxwell;
        BlockVector<double> maxwell_system_rhs;
        BlockVector<double> maxwell_load_vector;

        // matrices for ns solver
        SparsityPattern      velocity_sparsity_pattern;
        SparsityPattern      pressure_sparsity_pattern;
        SparsityPattern      vel_pres_sparsity_pattern;

        SparseMatrix<double> velocity_laplace_matrix;
        SparseMatrix<double> pressure_laplace_matrix;

        SparseMatrix<double> velocity_mass_matrix;
        SparseMatrix<double> pressure_mass_matrix;

        SparseMatrix<double> advection_matrix;
        SparseMatrix<double> gradient_matrix;

        SparseMatrix<double> velocity_step_matrix_const;
        SparseMatrix<double> velocity_step_matrix;
        SparseMatrix<double> pressure_step_matrix;

        // vectors for ns solver
        Vector<double> lorenz_force;
        Vector<double> diffusion_rhs;
        Vector<double> projection_rhs;

        Vector<double> previous_velocity;
        Vector<double> current_velocity;
        Vector<double> previous_pressure;
        Vector<double> current_pressure;
        Vector<double> previous_phi;
        Vector<double> current_phi;

        Vector<double> u_star;
        Vector<double> p_star;

        // preconditioners for ns solver
        SparseDirectUMFPACK pressure_mass_inv;
        SparseDirectUMFPACK pressure_laplace_inv;
        SparseILU<double>   diffusion_preconditioner;
        SparseILU<double>   projection_preconditioner;

        // exact solutions (used for BCs too)
        ExactMaxwell<dim>   exact_maxwell;
        ExactVelocity<dim> exact_velocity;
        ExactPressure<dim> exact_pressure;

};

// MHD constructor
template <int dim>
MHD<dim>::MHD(const unsigned int degree,
              const unsigned int n_global_refinements)
    : degree(degree)
    , n_global_refinements(n_global_refinements)
    , time_step(std::pow(0.1, n_global_refinements))
    , end_time(1.0)
    , current_time(0.0)
    , time_step_number(1)
    , nu(1.0)
    , beta(1.0)
    , eta(0.1)
    , mu(1.0)
    , rho(1.0)
    , standard(false)
    , fe_maxwell(FE_NedelecSZ<dim>(degree+1), 1, FE_Q<dim>(degree), 1)
    , fe_velocity(FE_Q<dim>(degree+1), dim)
    , fe_pressure(degree)
    , maxwell_dof_handler(triangulation)
    , velocity_dof_handler(triangulation)
    , pressure_dof_handler(triangulation)
{}

// set up the grid - basic hyper cube with specified refinement
template <int dim>
void
MHD<dim>::setup_grid()
{
  GridGenerator::hyper_cube(triangulation);

  triangulation.refine_global(n_global_refinements);

  // maxwell can't handle faster time stepping but ns can
  //double dx = GridTools::minimal_cell_diameter(triangulation);
  // time_step = dx; // this will just speed things up
}

// setup dofs for maxwell system
// separate magnetic and lagrange components
// to ensure system has block structure
template <int dim>
void MHD<dim>::setup_maxwell_dofs()
{
    // make sure system is clear
    maxwell_system_matrix.clear();
    maxwell_unconstrained_mass_matrix.clear();
    maxwell_unconstrained_system_matrix.clear();
    maxwell_preconditioner_matrix.clear();
    maxwell_lagrange_mass_matrix.clear();

    maxwell_dof_handler.distribute_dofs(fe_maxwell);

    // group together the velocity components
    // separate from the lagrange multiplier
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;

    DoFRenumbering::component_wise(maxwell_dof_handler);

    std::vector<types::global_dof_index> dofs_per_component(
            dim+1, types::global_dof_index(0));

    DoFTools::count_dofs_per_component(maxwell_dof_handler,
                                       dofs_per_component,
               true); // this means there will be no dublicates

    const unsigned int n_h = dofs_per_component[0];
    const unsigned int n_q = dofs_per_component[dim];

    // it is overkill to use exact solution here
    exact_maxwell.set_time(current_time);

    {
      maxwell_constraints.clear();
      DoFTools::make_hanging_node_constraints(maxwell_dof_handler, 
                                              maxwell_constraints);

      // FE_Nedelec boundary condition.
      VectorTools::project_boundary_values_curl_conforming_l2(
          maxwell_dof_handler,
          0,
          exact_maxwell,
          0,
          maxwell_constraints,
          StaticMappingQ1<dim>::mapping);

      FEValuesExtractors::Scalar q(dim);

      // Lagrange multiplier boundary conditions
      VectorTools::interpolate_boundary_values(maxwell_dof_handler,
                                               0,
                                               exact_maxwell,
                                               maxwell_constraints,
                                               fe_maxwell.component_mask(q));
    }
    maxwell_constraints.close();

    // setup sparsity patterns using constraints
    {
        BlockDynamicSparsityPattern dsp(2, 2);
        dsp.block(0, 0).reinit(n_h, n_h);
        dsp.block(1, 0).reinit(n_q, n_h);
        dsp.block(0, 1).reinit(n_h, n_q);
        dsp.block(1, 1).reinit(n_q, n_q);

        dsp.collect_sizes();

        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);

        DoFTools::make_sparsity_pattern(
          maxwell_dof_handler, dsp, maxwell_constraints, false);

        maxwell_sparsity_pattern.copy_from(dsp);
    }

    {
        BlockDynamicSparsityPattern unconstrained_dsp(2, 2);
        unconstrained_dsp.block(0, 0).reinit(n_h, n_h);
        unconstrained_dsp.block(1, 0).reinit(n_q, n_h);
        unconstrained_dsp.block(0, 1).reinit(n_h, n_q);
        unconstrained_dsp.block(1, 1).reinit(n_q, n_q);

        unconstrained_dsp.collect_sizes();

        DoFTools::make_sparsity_pattern(maxwell_dof_handler,
                                        unconstrained_dsp);
        maxwell_unconstrained_sparsity_pattern.copy_from(unconstrained_dsp);
    }


    {
        BlockDynamicSparsityPattern preconditioner_dsp(2, 2);
        preconditioner_dsp.block(0, 0).reinit(n_h, n_h);
        preconditioner_dsp.block(1, 0).reinit(n_q, n_h);
        preconditioner_dsp.block(0, 1).reinit(n_h, n_q);
        preconditioner_dsp.block(1, 1).reinit(n_q, n_q);

        preconditioner_dsp.collect_sizes();

        DoFTools::make_sparsity_pattern(maxwell_dof_handler,
                                      preconditioner_dsp,
                                      maxwell_constraints,
                                      false);

        maxwell_preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
    }

    // initialize matrices with sparsity patterns
    maxwell_system_matrix.reinit(maxwell_sparsity_pattern);
    maxwell_preconditioner_matrix.reinit(
            maxwell_preconditioner_sparsity_pattern);
    maxwell_lagrange_mass_matrix.reinit(
            maxwell_preconditioner_sparsity_pattern.block(1,1));
    maxwell_unconstrained_mass_matrix.reinit(
            maxwell_unconstrained_sparsity_pattern);
    maxwell_unconstrained_system_matrix.reinit(
            maxwell_unconstrained_sparsity_pattern);

    // initialize vectors with sparsity patterns
    previous_maxwell.reinit(2);
    previous_maxwell.block(0).reinit(n_h);
    previous_maxwell.block(1).reinit(n_q);
    previous_maxwell.collect_sizes();

    current_maxwell.reinit(2);
    current_maxwell.block(0).reinit(n_h);
    current_maxwell.block(1).reinit(n_q);
    current_maxwell.collect_sizes();

    maxwell_system_rhs.reinit(2);
    maxwell_system_rhs.block(0).reinit(n_h);
    maxwell_system_rhs.block(1).reinit(n_q);
    maxwell_system_rhs.collect_sizes();

    maxwell_load_vector.reinit(2);
    maxwell_load_vector.block(0).reinit(n_h);
    maxwell_load_vector.block(1).reinit(n_q);
    maxwell_load_vector.collect_sizes();

    // use exact solution to set up initial condition
    VectorTools::project(maxwell_dof_handler,
                         maxwell_constraints,
                         QGauss<dim>(degree + 2),
                         exact_maxwell,
                         current_maxwell);

}

// setup dofs for ns system no need to
// worry about block structure here
template <int dim>
void MHD<dim>::setup_ns_dofs()
{
    // clear everything
    velocity_laplace_matrix.clear();
    pressure_laplace_matrix.clear();
    velocity_mass_matrix.clear();
    pressure_mass_matrix.clear();
    advection_matrix.clear();
    gradient_matrix.clear();

    // distribute all dofs
    velocity_dof_handler.distribute_dofs(fe_velocity);
    DoFRenumbering::Cuthill_McKee(velocity_dof_handler);

    pressure_dof_handler.distribute_dofs(fe_pressure);
    DoFRenumbering::Cuthill_McKee(pressure_dof_handler);

    // initialize all vectors
    previous_velocity.reinit(velocity_dof_handler.n_dofs());
    previous_pressure.reinit(pressure_dof_handler.n_dofs());
    previous_phi.reinit(pressure_dof_handler.n_dofs());

    current_velocity.reinit(velocity_dof_handler.n_dofs());
    current_pressure.reinit(pressure_dof_handler.n_dofs());
    current_phi.reinit(pressure_dof_handler.n_dofs());

    u_star.reinit(velocity_dof_handler.n_dofs());
    p_star.reinit(pressure_dof_handler.n_dofs());

    lorenz_force.reinit(velocity_dof_handler.n_dofs());
    diffusion_rhs.reinit(velocity_dof_handler.n_dofs());
    projection_rhs.reinit(pressure_dof_handler.n_dofs());

    // overkill to use exact solution here
    exact_velocity.set_time(current_time);
    exact_pressure.set_time(current_time);

    VectorTools::interpolate_boundary_values(
            velocity_dof_handler, 0, exact_velocity, velocity_constraints);
    velocity_constraints.close();

    VectorTools::interpolate_boundary_values(
            pressure_dof_handler, 0, exact_pressure, pressure_constraints);
    pressure_constraints.close();
}

// assemble the maxwell system
// some matrices will change in time and others wont
// so the boolean determines which to update
template <int dim>
void MHD<dim>::assemble_maxwell_system(bool initial)
{
    // clear matrices
    maxwell_system_matrix               = 0;
    maxwell_unconstrained_system_matrix = 0;
    if (initial)
    {
        maxwell_preconditioner_matrix     = 0;
        maxwell_lagrange_mass_matrix      = 0;
        maxwell_unconstrained_mass_matrix = 0;
    }

    QGauss<dim> quadrature_formula(degree+2);

    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping,
                            fe_maxwell,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);

    FEValues<dim> fe_values_vel(mapping,
                                fe_velocity,
                                quadrature_formula,
                                update_values | update_quadrature_points);
    FEValuesViews::Vector<dim> fe_views_vel(fe_values_vel, 0);


    const unsigned int dofs_per_cell = fe_maxwell.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_system_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_preconditioner_matrix(dofs_per_cell, dofs_per_cell);

    Tensor<1, dim>              val_i_h, val_j_h;
    double                      val_i_q, val_j_q;
    double                      curl_i_h, curl_j_h;
    double                      vel_cross_val_j_h;
    double                      div_i_h, div_j_h;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<Tensor<1,dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector h(0);
    const FEValuesExtractors::Scalar q(dim);

    // iterate through maxwell and velocity cells simultaneously
    // necessary because this is where the couping occurs
    auto cell_m = maxwell_dof_handler.begin_active();
    const auto cell_end_m = maxwell_dof_handler.end();
    auto cell_v = velocity_dof_handler.begin_active();
    while (cell_m != cell_end_m)
    {
      Assert(cell_m->center() == cell_v->center(),
             ExcMessage("a real bad thing happened"));

      cell_mass_matrix = 0;
      cell_system_matrix = 0;
      if (initial)
        cell_preconditioner_matrix = 0;
      fe_values.reinit(cell_m);
      fe_values_vel.reinit(cell_v);

      // extract velocity values
      fe_views_vel.get_function_values(current_velocity, velocity_values);
      for (unsigned int q_index = 0; q_index < n_q_points;
             ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              val_i_h = fe_values[h].value(i, q_index);
              val_i_q = fe_values[q].value(i, q_index);
              curl_i_h = fe_values[h].curl(i, q_index)[0];
              div_i_h  = fe_values[h].divergence(i,q_index);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  val_j_h = fe_values[h].value(j, q_index);
                  val_j_q = fe_values[q].value(j, q_index);
                  curl_j_h = fe_values[h].curl(j,q_index)[0];
                  div_j_h  = fe_values[h].divergence(j,q_index);

                  if (initial)
                  {
                      // pressure mass matrix aka preconditioner
                      cell_preconditioner_matrix(i, j) +=
                            val_i_q * val_j_q *
                            fe_values.JxW(q_index);
                  }
                  // mass matrix
                  cell_mass_matrix(i, j) +=
                        val_i_h * val_j_h *
                        fe_values.JxW(q_index);

                  // u cross h
                  vel_cross_val_j_h =
                      velocity_values[q_index][0] * val_j_h[1] -
                      velocity_values[q_index][1] * val_j_h[0];

                  // system matrix
                  cell_system_matrix(i,j) +=
                      ( nu * curl_i_h * curl_j_h -
                        curl_i_h * vel_cross_val_j_h
                        +
                        beta * val_i_h[0] * div_j_h
                        -
                        div_i_h * val_j_q -
                        val_i_q * div_j_h
                        ) * fe_values.JxW(q_index);

                }
            }
        }

      cell_system_matrix *= time_step;
      cell_system_matrix.add(1.0, cell_mass_matrix);

      cell_m->get_dof_indices(local_dof_indices);

      // distribute constraints
      maxwell_constraints.distribute_local_to_global(
        cell_system_matrix, local_dof_indices, maxwell_system_matrix);
      maxwell_unconstrained_system_matrix.add(local_dof_indices, cell_system_matrix);

      if (initial)
      {
          maxwell_constraints.distribute_local_to_global(
            cell_preconditioner_matrix, local_dof_indices, maxwell_preconditioner_matrix);

          maxwell_unconstrained_mass_matrix.add(local_dof_indices, cell_mass_matrix);
      }
      
      // iterate the cells
      ++cell_v;
      ++cell_m;

    }
    // store the lagrange mass matrix for preconditioning the outer solver
    if (initial)
        maxwell_lagrange_mass_matrix.copy_from(maxwell_preconditioner_matrix.block(1,1));
}

// solves the maxwell system one time step
// boolean determines if system matrices are updated
template <int dim>
void MHD<dim>::solve_maxwell(bool update)
{
    if (update)
        assemble_maxwell_system(false);

    // The current solution should be swapped with the previous solution:
    std::swap(previous_maxwell, current_maxwell);

    // Set up M h^k + dt f^{k + 1}
    {
      VectorTools::create_right_hand_side(maxwell_dof_handler,
                                          QGauss<dim>(degree + 2),
                                          Functions::ZeroFunction<dim>(dim+1),
                                          maxwell_load_vector);
      maxwell_load_vector *= time_step;
      
      maxwell_unconstrained_mass_matrix.vmult_add(maxwell_load_vector, 
                                          previous_maxwell);
    }

    // setup constraints for imposing boundary conditions
    {
      maxwell_constraints.clear();
      exact_maxwell.set_time(current_time + time_step);
      DoFTools::make_hanging_node_constraints(maxwell_dof_handler, 
                                              maxwell_constraints);

      // FE_Nedelec boundary condition.
      VectorTools::project_boundary_values_curl_conforming_l2(
          maxwell_dof_handler,
          0,
          exact_maxwell,
          0,
          maxwell_constraints,
          StaticMappingQ1<dim>::mapping);

      FEValuesExtractors::Scalar q(dim);

      // Lagrange multilier boundary condition.
      VectorTools::interpolate_boundary_values(maxwell_dof_handler,
                                               0,
                                               exact_maxwell,
                                               maxwell_constraints,
                                               fe_maxwell.component_mask(q));
    }
    maxwell_constraints.close();

    // Now we want to set up C^T (b - A k)
    auto u_system_operator = block_operator(maxwell_unconstrained_system_matrix);
    auto setup_constrained_rhs = constrained_right_hand_side(
        maxwell_constraints, u_system_operator, maxwell_load_vector);

    setup_constrained_rhs.apply(maxwell_system_rhs);

    const auto &F = maxwell_system_rhs.block(0);

    // setup schur complement
    const auto &A = maxwell_system_matrix.block(0,0);
    const auto &B = maxwell_system_matrix.block(1,0);
    const auto &B_T = maxwell_system_matrix.block(0,1);
    Vector<double> tmp(maxwell_system_rhs.block(0).size());
    Vector<double> schur_rhs(maxwell_system_rhs.block(1).size());

    SparseDirectUMFPACK A_inv;
    A_inv.factorize(A);


    // outer preconditioning
    // set schur complement preconditioner as inverse pressure mass matrix
    PreconditionSSOR<SparseMatrix<double>> preconditioner_M;
    preconditioner_M.initialize(maxwell_lagrange_mass_matrix);
    InverseMatrix<SparseMatrix<double>,
                    PreconditionSSOR<SparseMatrix<double>> >
              preconditioner_S(
                  maxwell_lagrange_mass_matrix, preconditioner_M);

    // define schur complement
    SchurComplement<SparseDirectUMFPACK> schur_comp(
                  maxwell_system_matrix, A_inv);
    
    // define inverse operator of schur complement
    InverseMatrix<SchurComplement<SparseDirectUMFPACK>,
                    InverseMatrix<SparseMatrix<double>,
                    PreconditionSSOR<SparseMatrix<double>> > > S_inv(
                  schur_comp, preconditioner_S);

    // compute schur_rhs
    A_inv.vmult(tmp, F);
    B.vmult(schur_rhs, tmp);
    schur_rhs -= maxwell_system_rhs.block(1);

    // solve for q
    S_inv.vmult(current_maxwell.block(1), schur_rhs);
    maxwell_constraints.distribute(current_maxwell);

    // compute second system rhs
    B_T.vmult(tmp, current_maxwell.block(1));
    tmp *= -1;
    tmp += F;

    // solve for h
    A_inv.vmult(current_maxwell.block(0), tmp);
    maxwell_constraints.distribute(current_maxwell);
}

// assemble static ns matrices
// the one dynamic matrix (advection) is updated
// in its own function
template <int dim>
void MHD<dim>::assemble_ns_matrices()
{
    QGauss<dim> qf(degree+2);

    // velocity sparsity pattern
    DynamicSparsityPattern u_dsp(velocity_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(velocity_dof_handler,
                                    u_dsp,
                                    velocity_constraints,
                                    /* keep_constrained_dofs */ true);
    velocity_sparsity_pattern.copy_from(u_dsp);

    // initialize velocity matrices
    velocity_laplace_matrix.reinit(velocity_sparsity_pattern);
    velocity_mass_matrix.reinit(velocity_sparsity_pattern);
    advection_matrix.reinit(velocity_sparsity_pattern);
    velocity_step_matrix_const.reinit(velocity_sparsity_pattern);
    velocity_step_matrix.reinit(velocity_sparsity_pattern);

    // define mass matrix and stiffness matrix
    MatrixCreator::create_mass_matrix(velocity_dof_handler,
                                      qf,
                                      velocity_mass_matrix);

    MatrixCreator::create_laplace_matrix(velocity_dof_handler,
                                         qf,
                                         velocity_laplace_matrix);
    // define 3 rho / 2dt M + mu L
    velocity_step_matrix_const = 0.0;
    velocity_step_matrix_const.add(mu, velocity_laplace_matrix);
    velocity_step_matrix_const.add(1.5*rho/time_step, velocity_mass_matrix);

    // pressure sparsity pattern
    DynamicSparsityPattern p_dsp(pressure_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(pressure_dof_handler,
                                    p_dsp,
                                    pressure_constraints,
                                    /* keep_constrained_dofs */ true);
    pressure_sparsity_pattern.copy_from(p_dsp);

    // initialize pressure matrices
    pressure_laplace_matrix.reinit(pressure_sparsity_pattern);
    pressure_mass_matrix.reinit(pressure_sparsity_pattern);
    pressure_step_matrix.reinit(pressure_sparsity_pattern);

    // define mass matrix and stiffness matrix
    MatrixCreator::create_mass_matrix(pressure_dof_handler,
                                      qf,
                                      pressure_mass_matrix);
    MatrixCreator::create_laplace_matrix(pressure_dof_handler,
                                         qf,
                                         pressure_laplace_matrix);

    // velocity-pressure sparsity pattern
    DynamicSparsityPattern g_dsp(velocity_dof_handler.n_dofs(),
                                 pressure_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(velocity_dof_handler,
                                    pressure_dof_handler,
                                    g_dsp);
    vel_pres_sparsity_pattern.copy_from(g_dsp);

    // initialize gradient operator matrix
    gradient_matrix.reinit(vel_pres_sparsity_pattern);
    gradient_matrix = 0;

    FEValues<dim> fe_values_v(fe_velocity, qf,
                              update_gradients | update_JxW_values);
    FEValues<dim> fe_values_p(fe_pressure, qf,
                              update_values | update_JxW_values);

    FEValuesViews::Vector<dim> fe_views_v(fe_values_v, 0);

    const unsigned int dofs_per_cell_v = fe_velocity.n_dofs_per_cell();
    const unsigned int dofs_per_cell_p = fe_pressure.n_dofs_per_cell();
    const unsigned int n_q_points = qf.size();

    FullMatrix<double> cell_gradient_matrix(dofs_per_cell_v, dofs_per_cell_p);
    std::vector<types::global_dof_index> local_dof_indices_v(dofs_per_cell_v);
    std::vector<types::global_dof_index> local_dof_indices_p(dofs_per_cell_p);

    double div_i, val_j;

    // iterate through velocity cell and pressure cells simultaneously
    auto cell_v = velocity_dof_handler.begin_active();
    const auto cell_end_v = velocity_dof_handler.end();
    auto cell_p = pressure_dof_handler.begin_active();
    while (cell_v != cell_end_v)
    {
      Assert(cell_v->center() == cell_p->center(),
             ExcMessage("cells don't match!"));

      cell_gradient_matrix = 0;

      fe_values_v.reinit(cell_v);
      fe_values_p.reinit(cell_p);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell_v; ++i)
            {
              div_i = fe_views_v.divergence(i, q_index);

              for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
                {
                  val_j = fe_values_p.shape_value(j, q_index);

                                 cell_gradient_matrix(i,j) +=
                      -fe_values_v.JxW(q_index) * div_i * val_j;
                }
            }
        }
      
      // store in global matrix
      cell_v->get_dof_indices(local_dof_indices_v);
      cell_p->get_dof_indices(local_dof_indices_p);
      for (unsigned int i = 0; i < dofs_per_cell_v; ++i)
          for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
              gradient_matrix.add(local_dof_indices_v[i],
                                  local_dof_indices_p[j],
                                  cell_gradient_matrix(i,j));

      // iterate cells
      ++cell_v;
      ++cell_p;
    }
}

// initialize the navier stokes system
// define solutions for n=0 and n=1
template<int dim>
void MHD<dim>::initialize_ns_system()
{

    // use exact solution to set up initial condition
    VectorTools::interpolate(velocity_dof_handler,
                             exact_velocity,
                             previous_velocity);
    exact_velocity.advance_time(time_step);
    VectorTools::interpolate(velocity_dof_handler,
                             exact_velocity,
                             current_velocity);

    VectorTools::interpolate(pressure_dof_handler,
                             exact_pressure,
                             previous_pressure);
    exact_pressure.advance_time(time_step);
    VectorTools::interpolate(pressure_dof_handler,
                             exact_pressure,
                             current_pressure);
    previous_phi = 0.0;
    current_phi = 0.0;

    // define preconditioner
    projection_preconditioner.initialize(pressure_laplace_matrix);
    pressure_mass_inv.factorize(pressure_mass_matrix);
}

// assemble the advection matrix for ns
// this matrix is updated each time step
template <int dim>
void MHD<dim>::assemble_ns_advection_matrix()
{
    advection_matrix = 0;

    QGauss<dim> quadrature_formula(degree+2);

    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping,
                            fe_velocity,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);

    FEValuesViews::Vector<dim> fe_views(fe_values, 0);

    const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double>     cell_advection_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<Tensor<1,dim>> u_star_values(n_q_points);
    std::vector<Tensor<2,dim>> u_star_gradients(n_q_points);

    Tensor<1, dim>              val_i_u, val_j_u;
    Tensor<2, dim>              grad_j_u;
    double                      u_star_div;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : velocity_dof_handler.active_cell_iterators())
    {
      cell_advection_matrix = 0;

      fe_values.reinit(cell);

      // get values of u_star -- handles nonlinearity
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

      velocity_constraints.distribute_local_to_global(
        cell_advection_matrix, local_dof_indices, advection_matrix);

    }
}

// define the lorenz force
// based on magnetic field and updated each time step
template<int dim>
void MHD<dim>::assemble_lorenz_force()
{
    lorenz_force = 0;

    QGauss<dim> quadrature_formula(degree+2);

    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping,
                            fe_velocity,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);
    FEValuesViews::Vector<dim> fe_views(fe_values, 0);

    FEValues<dim> fe_values_mag(mapping,
                                fe_maxwell,
                                quadrature_formula, update_gradients |
                                update_values | update_quadrature_points);

    const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    Vector<double>     cell_lorenz(dofs_per_cell);

    Tensor<1, dim>              val_i_u;
    Tensor<1, dim>              grad_h_2;
    Tensor<1, dim>              h_div_grad_h;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // storage for relevant magnetic field quantities
    std::vector<Tensor<1,dim>> magnetic_values(n_q_points);
    std::vector<Tensor<2,dim>> magnetic_gradients(n_q_points);

    const FEValuesExtractors::Vector magnetic(0);

    // iterate through magnetic field and velocity cells together
    auto cell_v = velocity_dof_handler.begin_active();
    const auto cell_end_v = velocity_dof_handler.end();
    auto cell_m = maxwell_dof_handler.begin_active();
    while (cell_v != cell_end_v)
    {
      Assert(cell_v->center() == cell_m->center(),
             ExcMessage("cells don't match!"));

      cell_lorenz = 0;
      fe_values.reinit(cell_v);
      fe_values_mag.reinit(cell_m);

      // get magnetic field values
      fe_values_mag[magnetic].get_function_values(
              current_maxwell, magnetic_values);
      fe_values_mag[magnetic].get_function_gradients(
              current_maxwell, magnetic_gradients);
      for (unsigned int q_index = 0; q_index < n_q_points;
             ++q_index)
        {
          // define the lorenze force at the quadrature point
          grad_h_2[0] = magnetic_gradients[q_index][0][0] 
                        * magnetic_values[q_index][0]
                      + magnetic_gradients[q_index][1][0] 
                        * magnetic_values[q_index][1];
          grad_h_2[1] = magnetic_gradients[q_index][1][1] 
                        * magnetic_values[q_index][1]
                      + magnetic_gradients[q_index][0][1] 
                        * magnetic_values[q_index][0];

          h_div_grad_h = magnetic_gradients[q_index] * 
                         magnetic_values[q_index];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {

                val_i_u = fe_views.value(i, q_index);
                cell_lorenz(i) += 
                    val_i_u * (h_div_grad_h - grad_h_2) * fe_values.JxW(q_index);
            }
        }

      cell_v->get_dof_indices(local_dof_indices);

      velocity_constraints.distribute_local_to_global(
        cell_lorenz, local_dof_indices, lorenz_force);

      // iterate the cells
      ++cell_v;
      ++cell_m;

    }
}

// solve diffusion equation for updated velocity
template<int dim>
void MHD<dim>::ns_diffusion_step()
{
    // define u *
    u_star.equ(2.0, current_velocity);
    u_star -= previous_velocity;

    // define p #
    p_star.equ(-1.0, current_pressure);
    p_star.add(-4.0/3.0, current_phi, 1.0/3.0, previous_phi);
    //p_star *= -1.0;

    assemble_ns_advection_matrix();

    // define (v, f^k+1)
    assemble_lorenz_force();
    diffusion_rhs.equ(1.0, lorenz_force);

    // add 1/2dt (v, 4uk - uk-1)
    Vector<double> tmp(velocity_dof_handler.n_dofs());
    tmp.equ(2.0*rho/time_step, current_velocity);
    tmp.add(-0.5*rho/time_step, previous_velocity);
    velocity_mass_matrix.vmult_add(diffusion_rhs, tmp);

    // add (div v, p #)
    gradient_matrix.vmult_add(diffusion_rhs, p_star);

    // set u_n-1 = u_n
    previous_velocity = current_velocity;

    // define 3/2dt M + A + nu K
    velocity_step_matrix.copy_from(velocity_step_matrix_const);
    velocity_step_matrix.add(rho, advection_matrix);


    // boundary conditions
    exact_velocity.set_time(current_time+time_step);
    std::map<types::global_dof_index, double> boundary_values_u;
    VectorTools::interpolate_boundary_values(velocity_dof_handler,
                                             0,
                                             exact_velocity,
                                             boundary_values_u);
    MatrixTools::apply_boundary_values(boundary_values_u,
                                       velocity_step_matrix,
                                       current_velocity,
                                       diffusion_rhs);

    // initialize preconditioner
    diffusion_preconditioner.initialize(velocity_step_matrix);

    // solver
    SolverControl solver_control(1000, 1e-8*diffusion_rhs.l2_norm());
    SolverGMRES<Vector<double>> gmres(solver_control);
    gmres.solve(velocity_step_matrix, current_velocity,
                diffusion_rhs, diffusion_preconditioner);
}

// solve projection step for auxillary variable phi
template<int dim>
void MHD<dim>::ns_projection_step()
{
    // get laplace matrix
    pressure_step_matrix.copy_from(pressure_laplace_matrix);

    projection_rhs = 0.0;
    // define projection right hand side
    gradient_matrix.Tvmult_add(projection_rhs, current_velocity);

    // iterate phi
    previous_phi = current_phi;

    // boundary condition.
    static std::map<types::global_dof_index, double> boundary_values_phi;
    VectorTools::interpolate_boundary_values(pressure_dof_handler,
                                         0,
                                         Functions::ZeroFunction<dim>(),
                                         boundary_values_phi);
    MatrixTools::apply_boundary_values(boundary_values_phi,
                                       pressure_step_matrix,
                                       current_phi,
                                       projection_rhs);

    // solver
    SolverControl solver_control(1000, 1e-8*projection_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    cg.solve(pressure_step_matrix, current_phi,
             projection_rhs, projection_preconditioner);
    current_phi *= 1.5/time_step;
}

// update pressure
template<int dim>
void MHD<dim>::ns_pressure_step()
{

    // set p_n-1 = p_n
    previous_pressure = current_pressure;

    if (standard)
        current_pressure += current_phi;
    else
    {
        current_pressure = projection_rhs;
        pressure_mass_inv.solve(current_pressure);
        current_pressure.sadd(1.0*mu, 1.0, previous_pressure);
        current_pressure += current_phi;
    }
}

// solve ns system using projection method
template <int dim>
void MHD<dim>::solve_ns()
{
    ns_diffusion_step();
    ns_projection_step();
    ns_pressure_step();
}


// iterate through each time step solving both systems
template <int dim>
void MHD<dim>::run()
{
    setup_grid();

    setup_maxwell_dofs();
    setup_ns_dofs();

    std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells() << std::endl;
    std::cout << "   Number of degrees of freedom: " 
        << velocity_dof_handler.n_dofs()
         + pressure_dof_handler.n_dofs() 
         + maxwell_dof_handler.n_dofs()
            << '\n'
            << "   Timestep size: "
            << time_step
            << std::endl;

    assemble_maxwell_system(true); // initialize everything

    assemble_ns_matrices();
    initialize_ns_system();

    for (; current_time <= end_time; current_time += time_step, ++time_step_number)
    {
        // solve for h^1 and h^2 without updating velocity field
        solve_maxwell(time_step_number > 2); // this boolean = don't update velocity

        // solve for u^k and p^k for all k >= 2
        if (time_step_number > 1)
            solve_ns();
    }
    output_results();
}

// output results to file
template <int dim>
void MHD<dim>::output_results()
{
    // Compute the pointwise maximum error:
    Vector<double> max_error_per_cell_m(triangulation.n_active_cells());
    {
        exact_maxwell.set_time(current_time);

        MappingQGeneric<dim> mapping(1);
        VectorTools::integrate_difference(mapping,
                                          maxwell_dof_handler,
                                          current_maxwell,
                                          exact_maxwell,
                                          max_error_per_cell_m,
                                          QIterated<dim>(QGauss<1>(2), 2),
                                          VectorTools::NormType::Linfty_norm);
        std::cout << "\t\t\t\tmaximum m error = "
                  << *std::max_element(max_error_per_cell_m.begin(),
                                       max_error_per_cell_m.end())
                  << std::endl;
    }

    // Compute the pointwise maximum error:
    Vector<double> max_error_per_cell_u(triangulation.n_active_cells());
    {
        exact_velocity.set_time(current_time);

        MappingQGeneric<dim> mapping(1);
        VectorTools::integrate_difference(mapping,
                                          velocity_dof_handler,
                                          current_velocity,
                                          exact_velocity,
                                          max_error_per_cell_u,
                                          QIterated<dim>(QGauss<1>(2), 2),
                                          VectorTools::NormType::Linfty_norm);
        std::cout << "\t\t\t\tmaximum u error = "
                  << *std::max_element(max_error_per_cell_u.begin(),
                                       max_error_per_cell_u.end())
                  << std::endl;
    }
    Vector<double> max_error_per_cell_p(triangulation.n_active_cells());
    {
        exact_pressure.set_time(current_time);

        MappingQGeneric<dim> mapping(1);
        VectorTools::integrate_difference(mapping,
                                          pressure_dof_handler,
                                          current_pressure,
                                          exact_pressure,
                                          max_error_per_cell_p,
                                          QIterated<dim>(QGauss<1>(2), 2),
                                          VectorTools::NormType::Linfty_norm);
        std::cout << "\t\t\t\tmaximum p error = "
                  << *std::max_element(max_error_per_cell_p.begin(),
                                       max_error_per_cell_p.end())
                  << std::endl;
    }

    // join the two systems
    const FESystem<dim> joint_fe(fe_maxwell, 1, fe_velocity, 1, fe_pressure, 1);
    DoFHandler<dim> joint_dof_handler(triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);
    Assert(joint_dof_handler.n_dofs() ==
           (maxwell_dof_handler.n_dofs() +
            velocity_dof_handler.n_dofs() +
            pressure_dof_handler.n_dofs()),
            ExcInternalError());

    Vector<double> joint_solution(joint_dof_handler.n_dofs());
    std::vector<types::global_dof_index>
      loc_joint_dof_indices(joint_fe.n_dofs_per_cell()),
      loc_max_dof_indices(fe_maxwell.n_dofs_per_cell()),
      loc_vel_dof_indices(fe_velocity.n_dofs_per_cell()),
      loc_pres_dof_indices(fe_pressure.n_dofs_per_cell());

    typename DoFHandler<dim>::active_cell_iterator
      joint_cell = joint_dof_handler.begin_active(),
      joint_endc = joint_dof_handler.end(),
      max_cell   = maxwell_dof_handler.begin_active(),
      vel_cell   = velocity_dof_handler.begin_active(),
      pres_cell  = pressure_dof_handler.begin_active();
    for (; joint_cell != joint_endc; ++joint_cell, ++max_cell, ++vel_cell, ++pres_cell)
      {
        joint_cell->get_dof_indices(loc_joint_dof_indices);
        max_cell->get_dof_indices(loc_max_dof_indices);
        vel_cell->get_dof_indices(loc_vel_dof_indices);
        pres_cell->get_dof_indices(loc_pres_dof_indices);
        for (unsigned int i = 0; i < joint_fe.n_dofs_per_cell(); ++i)
            if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                    joint_solution(loc_joint_dof_indices[i]) = current_maxwell(
                        loc_max_dof_indices[joint_fe.system_to_base_index(i)
                                          .second]);
                }
            else if (joint_fe.system_to_base_index(i).first.first == 1)
                {
                    joint_solution(loc_joint_dof_indices[i]) = current_velocity(
                        loc_vel_dof_indices[joint_fe.system_to_base_index(i)
                                          .second]);
                }
            else
                {
                    joint_solution(loc_joint_dof_indices[i]) = current_pressure(
                        loc_pres_dof_indices[joint_fe.system_to_base_index(i)
                                          .second]);
                }
      }

    std::vector<std::string> joint_solution_names(dim, "magnetic_field");
    joint_solution_names.emplace_back("lagrange_multiplier");
    joint_solution_names.emplace_back("velocity");
    joint_solution_names.emplace_back("velocity");
    joint_solution_names.emplace_back("pressure");

    // Save the output:
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(joint_dof_handler);

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          component_interpretation(
            2*dim + 2, DataComponentInterpretation::component_is_part_of_vector);
        component_interpretation[dim] =
          DataComponentInterpretation::component_is_scalar;
        component_interpretation[2*dim+1] =
          DataComponentInterpretation::component_is_scalar;

        data_out.add_data_vector(joint_solution,
                                 joint_solution_names,
                                 DataOut<dim>::type_dof_data,
                                 component_interpretation);
        data_out.add_data_vector(max_error_per_cell_m, "max_error_per_cell_m");
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

  for (unsigned int i = 0; i < 5; ++i)
    {
      MHD<2> mhd(degree, i);
      mhd.run();
    }

  return 0;
}


